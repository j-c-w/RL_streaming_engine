import os
import pickle
from queue import Queue
import time
import argparse
import logging
from collections import deque
from sim_anneal_tune import AnnealerPPO

import dgl
import torch
from util import get_graph_json, create_graph, output_json, print_graph
from preproc import PreInput
import numpy as np
from coolname import generate_slug
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from multiprocessing import Pool
import multiprocessing as mp

from envs.streaming_engine_env import StreamingEngineEnv
from ppo_discrete import PPO

import train

SETTINGS = {
    # Which tile to place things in --- 'First' means 
    # use the first tile available within the placement mask.
    "PlacementMode": "Random"
}
ANNEALER_CONFIG = {
    "ItersSinceImprovement": 10,
    "AnnealingRateStart": 10.0,
    "AnnealingRateReduction": .5,
    "AnnealingReductionInterval": 10, # (cycles)
    "MaxIters": 5000
}

def get_first_location(mask, device):
    for spoke_idx in range(device.spoke_count):
        for tile_idx in range(device.tile_count):
            if mask[tile_idx*device.spoke_count + spoke_idx] != 0:
                return tile_idx, spoke_idx
    return None, None

def get_random_location(mask, device, rng):
    valid_slots = []
    for spoke_idx in range(device.spoke_count):
        for tile_idx in range(device.tile_count):
            if mask[tile_idx*device.spoke_count + spoke_idx] != 0:
                valid_slots.append((tile_idx, spoke_idx))
    return rng.choice(valid_slots)


class SimulatedAnnealingException(Exception):
    pass

class SimulatedAnnealingEngineEnv(StreamingEngineEnv):
    def __init__(self, args, graph, tile_count, spoke_count, pipeline_depth, random):
        self.random = random
        self.graphdef = graph
        self.args = args

        self.steps_since_successful = 0
        self.last_was_do = False

        super().__init__(args, graph, tile_count, spoke_count, pipeline_depth)

    def reset(self):
        super().reset()
        # If we need to restart the same way every time, then do so.
        if self.args.same_starting_point:
            print("Reseting the random to seed", args.seed)
            self.random = random.Random(args.seed)
        self.initial_placement()
        print("Initail Placement done, have initial score of ", self.get_overall_time())
        return self.get_state(10.0)

    def get_state(self, annealing_rate):
        s = list(self.se.get_state())
        s.append(annealing_rate)
        s.append(self.get_overall_time())
        s.append(self.steps_since_successful)
        return s

    def step(self, action, annealing_rate, mode):
        node, new_idx, new_spoke_idx = action
        if mode == 'Do':
            if self.last_was_do:
                self.steps_since_successful = 0
            self.last_was_do = True
            self.steps_since_successful += 1
        elif mode == 'Undo':
            self.last_was_do = False

        # Get current location of node
        current_idx, current_spoke_idx = self.location_of(node)
        self.se.tiles[current_idx].unplace(current_spoke_idx)

        # Re-place
        self.se.tiles[new_idx].place(node, new_spoke_idx)
        # Propagate the ready times
        ready_time = self.compute_ready_time_from(node, new_idx, new_spoke_idx)
        # The current annealing rate is important to allow the NN to make
        # decisions about which moves to make.
        obs = self.get_state(annealing_rate)
        latency_time = self.get_overall_time()

        return obs, latency_time, False, {'ready_time': ready_time}

    # Get the overall latency of the schedule for evaluation.
    # TODO -- get II
    def get_overall_time(self):
        max_rtime = 0
        for n in self.placed_nodes:
            this_rtime = self.placed_nodes[n]['ready_time']
            max_rtime = max(this_rtime, max_rtime)

        return max_rtime

    def compute_ready_time_from(self, node, idx, spoke):
        # Compute from the preds, then update the succs.
        ready_time, _ = super()._get_ready_time((node, idx, spoke))
        self.placed_nodes[node]['ready_time'] = ready_time
        self.placed_nodes[node]['tile_slice'] = (idx, spoke)
        for succ in super()._get_successors(node):
            # Compute the new ready time:
            s_idx, s_spoke = self.location_of(succ)
            rtime = self.compute_ready_time_from(succ, s_idx, s_spoke)
            self.placed_nodes[succ]['ready_time'] = rtime

        return ready_time

    def initial_placement(self):
        # Use a random placement strategy.
        # TODO -- Rely on the RL from micron to provide an initial
        # placement.

        # Iterate over topo sort of nodes.
        asc = dgl.topological_nodes_generator(self.graphdef['graph'])
        lnodes = [i.item() for t in asc for i in t]
        for this_node in lnodes:
            # Get the mask:
            mask = self.get_mask(this_node)
            if not mask.any():
                raise SimulatedAnnealingException("No Valid Placement Slots for {node}".format(node=this_node))
            if not self._predecessors_placed(this_node):
                raise SimulatedAnnealingException("Predeciesoorss not plaaced")
            if self.placed_nodes.get(this_node):
                raise SimulatedAnnealingException("Node already placed")

            if SETTINGS['PlacementMode'] == "First":
                tile_idx, spoke_idx = get_first_location(mask, self.se)
            elif SETTINGS['PlacementMode'] == "Random":
                tile_idx, spoke_idx = get_random_location(mask, self.se, self.random)
            else:
                raise SimulatedAnnealingException("Unknown placement mode {mode}".format(mode=SETTINGS['PlacementMode']))

            self.se.tiles[tile_idx].place(this_node, spoke_idx)
            ready_time, _ = self._get_ready_time((this_node, tile_idx, spoke_idx))
            self.placed_nodes[this_node] = {'tile_slice': (tile_idx, spoke_idx), 'ready_time': ready_time }

    def set_graph(self, graphdef):
        super().set_graph(graphdef)

        # Start by creating an initial placement for this.
        # self.initial_placement()

def true_annealer(env, graph, ppo=None, queue=None, agent_mode=None):
    print("Starting true annealer")
    queue = []
    iterating = True
    iter_count = 0
    last_improvement = 0
    annealing_rate = ANNEALER_CONFIG['AnnealingRateStart']
    print("Starting toposort")
    print(graph['graph'])
    asc = dgl.topological_nodes_generator(graph['graph'])
    print("Done toposort")
    # Iterate backwarsd over nodes
    lnodes = [i.item() for t in asc for i in t][::-1]

    # For the aided annealer to keep track of things between
    # placements.
    tobuff = None
    rewards = 0.0

    try:
        state = env.reset()
    except SimulatedAnnealingException as e:
        # We need to address the scalability concerns with the
        # initial placement somehow.
        print("Failed to get initial placement due to " + str(e))
        return 999999999999, []
    best_graph = env.get_overall_time()

    while iterating:
        iter_count += 1
        if iter_count % ANNEALER_CONFIG['AnnealingReductionInterval'] == 0:
            if agent_mode == 'AidedAnnealing':
                if tobuff != None:
                    # Add the rewards we got from the last action setup.
                    # TODO -- should we scale this by the reduction rate?
                    print("Added reward ", -rewards)
                    queue.append((tobuff, (-rewards / 1000.0), False))
                    rewards = 0.0
                state = env.get_state(annealing_rate)
                action, tobuff = ppo.select_action(state, graph)
                reduction_rate, done = action.squeeze()
                reduction_rate = abs(reduction_rate) # Allow the agent to increase
                # the annealing rate too.
                print("done is ", done)
                if random.random() < done or iter_count > ANNEALER_CONFIG['MaxIters']:
                    return_queue = []
                    queue.append((tobuff, -best_graph * 10, False))
                    for elem in queue:
                        return_queue.append(stringify(elem))
                    return best_graph, return_queue
                annealing_rate = reduction_rate
            else:
                reduction_rate = ANNEALER_CONFIG['AnnealingRateReduction']
                annealing_rate *= reduction_rate
            if annealing_rate > 10.0:
                annealing_rate = 10.0
            print("Reducing annealing rate (factor {factor}), now allowed to get a factor {worse} worse.  After {iter} cycles, have best graph of {best_graph}".format(factor=reduction_rate, worse=annealing_rate, iter=iter_count, best_graph=best_graph))

        skipped = 0
        for node in lnodes:
            valid_spots = env.get_mask(node)
            # Pikc a random valid spot
            if skipped > 10000:
                assert False# Not making any progres -- TODO -- ahndle this better.
            if not valid_spots.any():
                skipped += 1
                continue
            skipped = 0
            old_idx, old_spoke = env.location_of(node)
            if agent_mode == 'SpokePlacement':
                new_spot, tobuff = ppo.select_action(state, graph, node, valid_spots)
            else:
                new_spot = random.randint(0, len(valid_spots) - 1)
                while valid_spots[new_spot] == 0.0:
                    new_spot = random.randint(0, len(valid_spots) - 1)
            new_idx = new_spot // env.se.spoke_count
            new_spoke = new_spot % env.se.spoke_count
            action = (node, new_idx, new_spoke)
            state, reward, done, _ = env.step(action, annealing_rate, 'Do')
            latency = env.get_overall_time()

            if agent_mode == 'SpokePlacement':
                queue.append((tobuff, -reward, done))
                # ppo.add_buffer(tobuff, -reward, done)
            elif agent_mode == 'AidedAnnealing':
                rewards += reward

            if latency < best_graph:
                last_improvement = iter_count
                best_graph = latency
            elif random.random() * (1.0 + annealing_rate) > float(latency) / float(best_graph):
                last_improvement = iter_count
                best_graph = latency
            else:
                # Undo:
                undo_action = (node, old_idx, old_spoke)
                state, undo_reward, _, _ = env.step(undo_action, annealing_rate, 'Undo')
                if agent_mode == 'SpokePlacement':
                    # Punish agent again.
                    queue.append((tobuff, -reward, done))
                    # ppo.add_buffer(tobuff, -undo_reward, done)

        if agent_mode != 'AidedAnnealing' and iter_count - last_improvement > ANNEALER_CONFIG['ItersSinceImprovement']:
            print("After iter {iter} have gone {cycles} since an update".format(iter=iter_count, cycles=(iter_count - last_improvement)))
            break

    if agent_mode == 'SpokePlacement':
        # Mark as done in PPO.
        # ppo.add_buffer(tobuff, 0.0, True)
        # Punish depending on how bad the best graph is --- aim is to
        # dwarf the other pnishments received along the way.
        # TODO -- make this work with arbitrary-sized inputs.
        if queue is not None and len(queue) > 2000:
            # There are OOM errors when passing massive states
            # back and forth --- leave some elements out (at random)
            random.shuffle(queue)
            queue = queue[:2000]
            pass
        queue.append((tobuff, -best_graph, True))
    elif agent_mode == 'AidedAnnealing':
        print("Adding final reward of ", -best_graph)
        queue.append((tobuff, -best_graph * 10, True))

    print("Number of udpates is ", len(queue))
    return_queue = []
    for elem in queue:
        return_queue.append(stringify(elem))
    return best_graph, return_queue

def stringify(i):
    return pickle.dumps(i, 0)

def train_tuned_annealer(args, ppo, graph):
    # Make this smaller for the tuned annealer, since it gives giner-grained control.
    ANNEALER_CONFIG['AnnealingReductionInterval'] //= 3
    bufs = []
    # TODO -- support MP.
    env = SimulatedAnnealingEngineEnv(args, graph, tile_count = args.device_topology[0], spoke_count = args.device_topology[1], pipeline_depth = args.pipeline_depth, random=random.Random(random.randint(0, 1000000)))
    for epoch in range(1, args.epochs + 1):
        print("Starting new epoch {i}".format(i=epoch))
        if args.use_mp:
            PROC = 20
            with Pool(PROC) as p:
                if args.same_starting_point:
                    envs = [
                        SimulatedAnnealingEngineEnv(args, graph, tile_count = args.device_topology[0], spoke_count = args.device_topology[1], pipeline_depth = args.pipeline_depth, random=random.Random(args.seed))
                        for _ in range(PROC)
                    ]
                else:
                    envs = [
                        SimulatedAnnealingEngineEnv(args, graph, tile_count = args.device_topology[0], spoke_count = args.device_topology[1], pipeline_depth = args.pipeline_depth, random=random.Random(random.randint(0, 1000000)))
                        for _ in range(PROC)
                    ]
                for i in range(PROC):
                    envs[i].set_graph(graph)

                print("Launching starmap")
                scores = p.starmap(true_annealer, [(env, graph, ppo, None, 'AidedAnnealing') for env in envs])
                this_scores = []
                rsum = 0.0
                for score, update in scores:
                    this_buf = []
                    this_scores.append(score)
                    this_sum = 0.0
                    for u in update:
                        buffer,reward,done = pickle.loads(u)
                        rsum+=reward
                        this_buf.append((buffer, reward, done))
                    print("Score is {score}, total reward was {reward}".format(score=score, reward=rsum))
                    rsum += this_sum

                score = min(this_scores)
        else:
            score, updates = true_annealer(env, graph, ppo, None, 'AidedAnnealing')
            rsum = 0.0
            this_buf = []
            for u in updates:
                buffer,reward,done = pickle.loads(u)
                rsum+=reward
                print("Looking at reward", reward)
                this_buf.append((buffer, reward, done))
            print("Score is {score}, total reward was {reward}".format(score=score, reward=rsum))

        added = 0
        # Normalize the rewards:
        bufs.append(this_buf)
        print("Update timestep is ", epoch, "and", PROC, "and", args.update_timestep // PROC)
        print("Mod is ", epoch % (args.update_timestep // PROC))
        if epoch % (args.update_timestep // PROC) == 0:
            print("At iteration {i}, updating the PPO".format(i=i))
            norm_factors = []
            iterate_bufs = []
            for sub_buf in bufs:
                # Compute the normalization factor: we take it as the mean
                this_norm_factor = 0.0
                num = 0
                for _, r, done in sub_buf:
                    this_norm_factor += r
                    num += 1
                if num > 0:
                    norm_factors.append(this_norm_factor / float(num))
                    iterate_bufs += sub_buf
            norm_factor = np.mean(norm_factors)

            for buffer,reward,done in iterate_bufs:
                ppo.add_buffer(buffer, -reward / abs(norm_factor), done)
                added +=1
            print("Added {items} to the PPO training".format(items=added))

            # true_annealer(env, graph, ppo)
            if added > 0:
                ppo.update()
                print("Iteration done, Updated PPO")
            else:
                print("No updates to inlcude in the PPO")

        if epoch % args.log_interval == 0:
            # print("Episode {i} best score {t}".format(i=i, t=best_score))
            this_time = time.time()
            fname="aided_annealer/annealer_model_epoch_" + str(this_time) + ".pth"
            print("Saving in file ", fname)
            torch.save(ppo.policy.state_dict(), fname)


def train_rl_annealing(args, ppo, graph):
    reward_buf = deque(maxlen=100)
    reward_buf.append(0)
    start = time.time()
    time_step = 0
    best_ready_time = float('inf')
    best_score = 1000000000

    if args.use_mp:
        mp.set_start_method('spawn')
    else:
        env = SimulatedAnnealingEngineEnv(args, graph, tile_count = args.device_topology[0], spoke_count = args.device_topology[1], pipeline_depth = args.pipeline_depth, random=random.Random(random.randint(0, 1000000)))
        env.set_graph(graph)

    for i in range(1, args.epochs + 1):
        print("Starting new epoch, {i}".format(i=i))
        # env.reset()
        if args.use_mp:
            PROC = 20
            with Pool(PROC) as p:
                if args.same_starting_point:
                    envs = [
                        SimulatedAnnealingEngineEnv(args, graph, tile_count = args.device_topology[0], spoke_count = args.device_topology[1], pipeline_depth = args.pipeline_depth, random=random.Random(args.seed))
                        for _ in range(PROC)
                    ]
                else:
                    envs = [
                        SimulatedAnnealingEngineEnv(args, graph, tile_count = args.device_topology[0], spoke_count = args.device_topology[1], pipeline_depth = args.pipeline_depth, random=random.Random(random.randint(0, 1000000)))
                        for _ in range(PROC)
                    ]
                for i in range(PROC):
                    envs[i].set_graph(graph)

                m = mp.Manager()
                queue = m.Queue(maxsize=100000)
                scores = p.starmap(true_annealer, [(env, graph, ppo, None, 'SpokePlacement') for env in envs])
                this_scores = []
                updates = []
                for score, update in scores:
                    this_scores.append(score)
                    updates += update
                this_score = min(this_scores)

        else:
            queue = Queue()
            PROC = 1
            this_score, updates = true_annealer(env, graph, ppo, queue, 'SpokePlacement')

        for score in this_scores:
            print("Got result score", score)

        # best_score = min(best_score, this_score)
        added = 0
        # while not queue.empty():
        #     buffer, reward, done = queue.get()
        #     ppo.add_buffer(buffer, reward, done)
        #     added += 1
        for update in updates:
            buffer,reward,done = pickle.loads(update)
            ppo.add_buffer(buffer, reward, done)
            added +=1
        print("Added {items} to the PPO training".format(items=added))

        # true_annealer(env, graph, ppo)
        if added > 0:
            ppo.update()
            print("Iteration done, Updated PPO")
        else:
            print("No updates to inlcude in the PPO")

        if i % args.log_interval == 0:
            print("Episode {i} best score {t}".format(i=i, t=best_score))
            this_time = time.time()
            fname="annealer_model_epoch_" + str(this_time) + ".pth"
            print("Saving in file ", fname)
            torch.save(ppo.policy.state_dict(), fname)


def run_annealing(args, graph):
    args.device_topology = tuple(args.device_topology)
    args.nodes = graph['graph']

    if args.debug:
        print(graph)

    device = {}
    device['topology'] = args.device_topology
    device['action_dim'] = np.prod(args.device_topology)

    preproc = PreInput(args)
    graphdef = preproc.pre_graph(graph, device)

    # Do the simulated annealar
    if args.true_annealer:
        rng = random.Random(args.seed)
        env = SimulatedAnnealingEngineEnv(args, graphdef, tile_count = args.device_topology[0], spoke_count = args.device_topology[1], pipeline_depth = args.pipeline_depth, random=rng)
        true_annealer(env, graphdef)
    elif args.train_rl_annealer:
        ppo = PPO(args, graphdef, device, state_dim = args.device_topology[0] * args.device_topology[1])

        train_rl_annealing(args, ppo, graphdef)
    elif args.tuned_annealer:
        ppo = AnnealerPPO(args, graphdef, device, state_dim = args.device_topology[0] * args.device_topology[1])
        train_tuned_annealer(args, ppo, graphdef)

if __name__ == "__main__":
    args = train.get_args()
    print(args.__dict__)
    if args.use_mp:
        mp.set_start_method('spawn')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    graph_json = get_graph_json(args.input)
    graphdef = create_graph(graph_json)
    run_annealing(args, graphdef)
