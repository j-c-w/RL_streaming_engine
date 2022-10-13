import pickle
import time
import argparse
import os
import subprocess
import json
import sys

import torch
import torch.nn as nn

import numpy as np

from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

from or_gym.utils import create_env

import random
import multiprocessing as mp

import train
import gym
import gym.spaces as spaces
import uuid
random.seed(0)

# If true, then this schedules on a tile-level.  If false,
# this schedules on an operation level, where operations
# are the same set provided by the ones in the input tiles.
BY_TILE = False

class Program:
    def __init__(self):
        pass

Instructions = [
    "fadd",
    "fsub",
    "fmul",
    "fdiv",
    "fmod",
    "mod",
    "add",
    "sub",
    "mul",
    "sdiv",
    "udiv",
    "and",
    "or",
    "xor",
    "icmp",
    "br",
    "sext",
    "zext",
    "shl",
    "lshr",
    "ashr",
    "not",
    "load",
    "store"
]

class PlacementModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        model_args = model_config['custom_model_config']
        print(model_args)
        true_obs_space = model_args['true_obs_space'] if 'true_obs_space' in model_args else None
        rows = model_args['rows']
        cols = model_args['cols']
        with_features = model_args['with_features']
        cgra_state = not model_args['no_cgra_state']

        if true_obs_space is None:
            if cgra_state:
                true_obs_space = ((rows + 1) * (cols + 1) * len(Instructions),)
            else:
                # If we don't use the CGRA state, then we only care
                # about the instruction passed in
                true_obs_space = (len(Instructions), )
        if with_features:
            # Add the size of the observation space,
            # which is one entry per instruction, times
            # the granulatity of operations in that space.
            true_obs_space = (true_obs_space[0] + program_features_size(),)
            self.with_features = True
        print("Initializing placement model with ", true_obs_space)
        # TODO --- find a better way of passing this.  The 630 is a hack to get around the
        # mask being passed as part of obs_space.
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(
            spaces.Box(-1, 1, shape=true_obs_space), action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        actions = input_dict['obs']['action_mask']
        input_dict["obs"] = input_dict["obs"]['observation']
        # print("Looking at dict", input_dict['obs'].size())
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        mask = torch.max(torch.log(actions), torch.tensor(-1000.0))
        return fc_out + mask, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

def run_on_benchmark(args):
    benchmark, cgra, config_id = args
    id = str(uuid.uuid4())

    print("Starting run of benchmark", benchmark)
    # Don't really get why the flat with an arg has to go in quotes
    # here, there it is something a bit off with the nix-wrapper
    # that this has to be accessed through.
    with open(cgra.args.temp_folder + '/temp_output_' + id, 'w') as f:
        p = subprocess.Popen(('cgra-mapper', benchmark.file, '--params-file ~/Projects/CGRA/RL_streaming_engine/' + cgra.args.temp_folder + '/temp_config' + config_id + '.json' + ' --use-egraphs --egraph-mode frequency'), stderr=f, stdout=f)
        p.wait()
    # Have to deal with the weird gibberish that opt puts in the
    # stdout.  Probably faster with this anyway.
    with open(cgra.args.temp_folder + '/temp_output_filtered_' + id, 'w') as f:
        p = subprocess.Popen(('grep', '--text', '-e', 'II', cgra.args.temp_folder + '/temp_output_' + id), stdout=f)
        p.wait()
    found_II = False

    II = 100000 ## Large punishment II if the bencmark doesn't build
    with open(cgra.args.temp_folder + '/temp_output_filtered_' + id, 'r') as f:
        for line in f.readlines():
            if line.startswith('II'):
                II = int(line.split(':')[1].strip())
                found_II = True
                if found_II:
                    pass
    print("IIs found was", II)
    return II

class Benchmark(object):
    # TODO --- Add DFG for passing to Agent.
    def __init__(self, fname, args, prog_features=False, weight=1.0):
        self.file = fname
        if prog_features:
            # Extract the program features using CGRA-Mapper.
            basefname = os.path.basename(fname)
            p = subprocess.Popen(('cgra-mapper', self.file, '--features', cgra.args.temp_folder + '/features' + basefname + '.json'), stdout=subprocess.PIPE)
            p.wait()
            self.features = json.load(open(cgra.args.temp_folder + '/features' + basefname + '.json'))

    def __str__(self):
        return self.file

class CGRA(object):
    def __init__(self, benchmarks, rows, cols, args) -> None:
        self.programs = benchmarks
        self.rows = rows
        self.cols = cols
        self.args = args
        self.setup_placement()
        self.num_tiles = self.rows * self.cols
        self.num_operations = len(Instructions)

    def get_placement(self, i, j):
        return self.placements[i][j]

    def get_placement_from_index(self, i):
        row = i % self.rows
        col = i // self.rows

        return self.placements[row][col]

    def swap(self, from_ind, to_ind, op=None):
        from_row = from_ind % self.rows
        from_col = from_ind // self.rows

        to_row = to_ind % self.rows
        to_col = to_ind // self.rows

        if not BY_TILE:
            tmp_to = self.placements[to_row][to_col][op].float().item()
            tmp_from = self.placements[from_row][from_col][op].float().item()

            self.placements[from_row][from_col][op] = tmp_to
            self.placements[to_row][to_col][op] = tmp_from
        else:
            tmp_to = self.placements[to_row][to_col]
            tmp_from = self.placements[from_row][from_col]

            self.placements[from_row][from_col] = tmp_to
            self.placements[to_row][to_col] = tmp_from

    def set_placement(self, i, tile):
        r = i % self.rows
        c = i // self.rows

        self.placements[r][c] = tile

    # Add the operations in ops to the 
    # tile in cell i.
    def add_operation_to(self, i, ops):
        r = i % self.rows
        c = i // self.rows

        self.placements[r][c] += ops

    def setup_placement(self):
        self.placements = []
        for i in range(self.rows):
            p = []
            for j in range(self.cols):
                p.append(torch.zeros(size=[len(Instructions)]))
            self.placements.append(p)

    def reset(self):
        self.setup_placement()

    def compute_reward(self, parallel=False):
        cgra_config = self.create_cgra_mapper_config()
        # TODO --- parallelize.
        id = str(uuid.uuid4())
        with open(self.args.temp_folder + '/temp_config' + id + '.json', 'w') as f:
            f.write(json.dumps(cgra_config))

        # Run the CGRA Mapper for each of the benchmarks.
        if parallel:
            with mp.Pool(15) as p:
                IIs = p.map(run_on_benchmark, [(b, self, id) for b in self.programs])
        else:
            IIs = []
            for benchmark in self.programs:
                IIs.append(run_on_benchmark((benchmark, self, id)))

        print("Computed IIs ", IIs)
        # TODO --- is something other than raw sum better?
        return -sum(IIs)

    def placed(self, row, col):
        # Really want to check === 0, but FP issues.
        # Should be >= 1.0, since 1.0 is set in each index
        # that is active.
        return sum(self.placements[row][col]) > 0.01

    def get_instruction_names(self, row, col):
        inst_tensor = self.get_placement(row, col)
        names = []
        for i in range(inst_tensor.size()[0]):
            if inst_tensor[i] > 0.0001:
                names.append(Instructions[i])
        return names

    def create_cgra_mapper_config(self):
        operations_dict = {}
        for row in range(self.rows):
            rname = str(row)
            operations_dict[rname] = {}
            for col in range(self.cols):
                cname = str(col)
                operations_dict[rname][cname] = self.get_instruction_names(row, col)

        result = {}
        result['kernel'] = ""
        result['targetFunction'] = False
        result['targetNested'] = True
        result['targetLoopsID'] = [0, 1]
        result['doCGRAMapping'] = True
        result['row'] = self.rows
        result['column'] = self.cols
        result['precisionAware'] = True
        result['heterogeneity'] = True
        result['isTrimmedDemo'] = True
        result['heuristicMapping'] = True
        result['bypassConstraint'] = 8
        result['isStaticElasticCGRA'] = False
        result['ctrlMemConstraint'] = 200
        result['regConstraint'] = 64
        result['opLatency'] = {
            'store': 2
        }
        result['optPipelined'] = ['store']
        result['additionalFunc'] = {
            'store': [4]
        }
        result['operations'] = operations_dict

        return result


class PlacementEnv(gym.Env):
    def __init__(self, cgras, programs, rows, cols, args, with_features=False, cgra_state=True):
        super(PlacementEnv, self).__init__()

        self.rows = rows
        self.config_cgras = cgras
        self.cols = cols
        self.args = args

        self.current_cgra = 0
        self.placement = CGRA(programs, rows, cols, args)
        self.setup_tiles()

        self.action_space = spaces.Discrete(self.rows * self.cols)

        self.with_features = with_features
        self.cgra_state = cgra_state

        # Current placement and the next placement.
        if self.cgra_state:
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(low=0.0, high=1.0, shape=[self.rows + 1, self.cols + 1, len(Instructions)]),
                "action_mask": spaces.Box(0, 1, shape=(self.rows * self.cols,))
            })
        else:
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(low=0.0, high=1.0, shape=[1, 1, len(Instructions)]),
                "action_mask": spaces.Box(0, 1, shape=(self.rows * self.cols,))
            })
        self.programs = programs
        self.tile_index = 0
        self.sum_reward = None

    # Sets up the tiles for the current CGRA
    def setup_tiles(self):
        print("Looking at cgra ", self.current_cgra)
        if BY_TILE:
            self.tiles = self.config_cgras[self.current_cgra].tiles
        else:
            # Split the tileopts by each operation they support.
            self.tiles = []
            tileopts = self.config_cgras[self.current_cgra].tiles
            for opts in tileopts:
                for i in range(opts.size()[0]):
                    if opts[i].float().item() > 0.0005:
                        tile_tensor = torch.zeros(size=[len(Instructions)])
                        tile_tensor[i] = 1.0
                        self.tiles.append(tile_tensor)

    def reset(self):
        if self.sum_reward is not None:
            # This is not the first run.
            print("Total reward was: ", self.sum_reward)
            print("==============================")
        self.sum_reward = 0.0
        self.tile_index = 0
        self.placement.reset()
        # Increment the operations we are placing.
        if self.args.test:
            # If we are evaluating, go in a determinsitc order through
            # the CGRAs.
            self.current_cgra = self.current_cgra + 1
            if self.current_cgra >= len(self.config_cgras):
                self.current_cgra = 0
        else:
            self.current_cgra = random.randint(0, len(self.config_cgras) - 1)
        self.setup_tiles()
        return self.get_observations()
    
    def get_instruction_embedding(self, instr):
        return instr

    def get_observations(self):
        # Compute the box obvservation space: 0, 0 is the
        # to-be-placed element, the other 0-col indexes
        # are empty.  the placement so far is then one-indexed
        # from there.
        if self.cgra_state:
            observations = torch.zeros(size=[(self.rows + 1) * (self.cols + 1) * len(Instructions)])
            observations = observations.view((self.rows + 1, self.cols + 1, len(Instructions)))
            for i in range(1, self.rows + 1):
                for j in range(1, self.cols + 1):
                    embedding = self.get_instruction_embedding(self.placement.get_placement(i - 1, j - 1))
                    observations[i][j] = embedding
        else:
            observations = torch.zeros(len(Instructions))
            observations = observations.view((1, 1, len(Instructions)))

        if self.tile_index < len(self.tiles):
            observations[0][0] = self.get_instruction_embedding(self.tiles[self.tile_index])
            next_tile = self.tiles[self.tile_index]
        else:
            # This happens in final observations after all the tiles have been placed.
            next_tile = self.tiles[0]

        if self.with_features:
            distance_cdfs = []
            # Compute the features of the vector
            for benchmark in benchmarks:
                distances_cdf = benchmark.get_features()
                distance_cdfs.append(distances_cdf)

            distance_cdfs.append()

        return {
            'observation': np.array(observations),
            'action_mask': np.array(self.get_mask(next_tile))
        }

    def get_mask(self, tile_to_place):
        mask = []
        for i in range(self.rows * self.cols):
            row_idx = i % self.rows
            col_idx = i // self.rows
            if BY_TILE and self.placement.placed(row_idx, col_idx):
                mask.append(0.0)
            elif not BY_TILE:
                # If not by tile, allow each operation to be placed
                # individually in each slot.
                tile = self.placement.get_placement(row_idx, col_idx)
                place_index = tile_to_place.nonzero()[0][0]
                if tile[place_index] > 0.00005:
                    mask.append(0.0) # Slot already filled.
                else:
                    mask.append(1.0)
            else:
                mask.append(1.0)
        return torch.tensor(mask)

    def step(self, action):
        row_idx = action % self.rows
        col_idx = action // self.rows

        # TODO -- assert that we are not double-placing due to broken mask.
        self.placement.placements[row_idx][col_idx] += self.tiles[self.tile_index]

        self.tile_index += 1
        if self.tile_index == len(self.tiles):
            done = True
        else:
            done = False

        if done:
            reward = self.placement.compute_reward()
        else:
            reward = 0.0

        self.sum_reward += reward
        return self.get_observations(), reward, done, {}

    def render(self):
        print("Rendering requested")

    def close(self):
        print("Closing gym")

def create_tile_for(instructions):
    tile = torch.empty(len(Instructions))
    for i in range(len(Instructions)):
        if Instructions[i] in instructions:
            tile[i] = 1.0
        else:
            tile[i] = 0.0

    return tile

class Annelaer():
    def __init__(self, cgra_desc):
        self.cgra_desc = cgra_desc
        self.annealer_config = {
            "InitialRate": 10.0,
            "ReductionRate": 0.5,
            "ReductionFrequency": 20,
            "CyclesSinceImprovement": 5
        }
        self.tracker = {
            "Steps": 0
        }

    def anneal(self, cgra):
        current_rate = self.annealer_config['InitialRate']
        current_cost = abs(cgra.compute_reward(parallel=True))

        cycles_since_improvement = 0
        cycles = 0
        while cycles_since_improvement < self.annealer_config['CyclesSinceImprovement']:
            cycles_since_improvement += 1
            cycles += 1
            self.tracker['Steps'] += 1

            # Do a random swap:
            max_length = cgra.num_tiles - 1
            random_source = random.randint(0, max_length)
            random_target = random.randint(0, max_length)
            if BY_TILE:
                op = None
            else:
                op = random.randint(0, cgra.num_operations - 1)
                # Only try and move things that are actually relevant -- but not
                # all tiles have operations, so for those, we are OK
                # just using the existing tile.
                selected_tile = cgra.get_placement_from_index(random_source)
                if sum(selected_tile) > 0.0:
                    while selected_tile[op] < 0.0005:
                        op = random.randint(0, cgra.num_operations - 1)
                else:
                    # Entire tile is empty, nothing to swap.
                    pass

            cgra.swap(random_source, random_target, op=op)
            cost = abs(cgra.compute_reward(parallel=False))

            # Compute the current cost:
            if cost < current_cost or random.random() * (1.0 + current_rate) > float(cost) / float(current_cost):
                print("Swapped", cost, 'for ', current_cost)
                current_cost = cost
                cycles_since_improvement = 0
            else:
                # Undo that random move.
                cgra.swap(random_target, random_source, op=op)

            if cycles % self.annealer_config['ReductionFrequency'] == 0:
                current_rate = current_rate * self.annealer_config['ReductionRate']
                print("Reducing rate to ", current_rate, "best score", current_cost)

            if cycles_since_improvement > self.annealer_config['CyclesSinceImprovement']:
                break
        return current_cost, cgra.placements

def create_readable_output_for(instr):
    res = []
    for i in range(instr.size()):
        elem = instr[i]
        if elem > 0.9999:
            # Really looking for elem = 1.0
            res.append(Instructions[i])
    return res

def load_cgra_tiles_from_manual_distribution(file):
    print("Loading manual from file", file)

    with open(file) as f:
        data = json.load(f)

    row = data['row']
    col = data['column']
    max_tiles = data['num_ops']

    tiles = []
    distribution = {}
    # format of this dict is { op: <count> }
    for op in data:
        if op == 'row' or op == 'column' or op == 'num_ops':
            continue

        distribution[op] = data[op]

        for i in range(data[op]):
            tiles.append(op)
    return row, col, tiles, distribution, max_tiles



def load_cgra_tiles_from_file(file):
    # Using a file in the format of CGRA-Mapper, load the tiels
    # and sizes.
    print("Loading from file", file)
    with open(file) as f:
        data = json.load(f)
    
    row = data['row']
    col = data['column']
    ops = data['operations']

    return row, col, ops

class CGRADescription():
    # If tiles is none, randomly initialize rows*cols number of tiles.
    def __init__(self, rows, cols, tiles=None, required_ops=Instructions):
        self.rows = rows
        self.cols = cols

        if tiles is None:
            self.tiles = []

            # We create one of each operation to make is always possible
            # to get a score.  Get the list of required ops as input.
            for op in required_ops:
                self.tiles.append(create_tile_for([op]))

            for i in range(rows * cols):
                self.tiles.append(create_tile_for(random.sample(Instructions, 2)))
        else:
            self.tiles = [create_tile_for(x) for x in tiles]

    def __str__(self):
        # TODO -- properly print the tiles.
        return "Rows: " + str(self.rows) + ", Cols: " + str(self.cols) + ", Tiles:" + str(tiles)

    # Set the tiles available to this Description by gaussion distribution
    # around the number of tiles in a description dictionary.
    def get_gaussian_tiles(self, tile_counts):
        for instr in Instructions:
            pass
        assert False # TODO --- Implement

    # Get the number of tiles according to the exponential distribution.
    def get_exponential_tiles(self, gamma=0.5, distribution=None, max_tiles=None):
        # Distribution should be a map from Instruction->gamma for that
        # instruction.
        default_gamma = gamma

        # This distribution has mean 1/gamma and variance 1/gamma^2.
        # So this function produces len(Instructions) * 1/gamma operations
        # on average.
        setup = False # Keep setting up until we generate architectures
        # with fewer than max_tiles.
        while not setup:
            self.tiles = []
            tile_count = 0

            for instr in Instructions:
                if distribution is not None:
                    if instr in distribution:
                        gamma = distribution[instr]
                    else:
                        # If the instruction is not in the distribution,
                        # disable generation of that instruction.
                        gamma = 10.0
                number = int(np.round(np.random.exponential(scale=1.0/gamma)) + 1)
                tile_count += number

                for i in range(number):
                    self.tiles.append(create_tile_for([instr]))
            if tile_count != max_tiles:
                setup = False
            else:
                setup = True

def load_benchmarks_from_json(args, jsonfile):
    benchmarks = []
    with open(jsonfile) as f:
        bench = json.load(f)
        descr = bench['benchmarks']
        for d in descr:
            benchmark_file = d['file']
            weight = float(d['weight'])

            benchmarks.append(Benchmark(benchmark_file, args, weight=weight))
    return benchmarks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp-folder', dest='temp_folder', default='temp', help='Temp folder for the compilation output files')
    parser.add_argument('CGRA', help='CGRA description JSON')
    parser.add_argument('benchmark_files')
    parser.add_argument('exploration_model')
    parser.add_argument('--annealer', dest='annealer', default=False, action='store_true', help='Use annealer')
    parser.add_argument('--train-only', dest='train_only', default=False, action='store_true', help='Dont run forever: just keep going until this is trained')
    parser.add_argument('--save-model', dest='save_model', default=None, help='File to store the generated model into.')
    parser.add_argument('--random-cgras', dest='random_cgras', default=0, type=int, help='Generate a number of random CGRAs for training.')
    parser.add_argument('--restore', dest='restore', default=None, help='Restore a previous run.')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='Test a model')
    parser.add_argument('--model-with-program-features', dest='with_program_features', default=False, action='store_true', help='Use program features in classification')
    parser.add_argument('--model-no-cgra-state', dest='no_cgra_state', default=False, action='store_true', help='Use current CGRA schedule in classification')
    parser.add_argument('--print-cgras', dest='print_cgras', default=False, action='store_true', help='Print the CGRAs that are generated.')
    parser.add_argument('--manual-distribution', dest='manual_distribution', default=False, action='store_true', help="Use a mangual distribution for teh CGRA input --- note that this has a different json import format.")
    args = parser.parse_args()

    if not os.path.exists(args.temp_folder):
        os.mkdir(args.temp_folder)

    if args.random_cgras:
        if args.manual_distribution:
            rows, cols, tiles, distribution, max_tiles = load_cgra_tiles_from_manual_distribution(args.CGRA)
        else:
            rows, cols, tiles = load_cgra_tiles_from_file(args.CGRA)
            distribution = None # TODO -- should we load a base distribution from this?
            max_tiles = None
        print("Loaded sizes are ", rows, cols)
        # Generate a bunch of random CGRAs for training on.
        # TODO -- should treat the base cgra as the min possible number
        # of operations.
        num_cgras = args.random_cgras
        cgras = []
        random.seed(1)
        for i in range(num_cgras):
            descr = CGRADescription(rows, cols, tiles=tiles)
            if args.exploration_model == 'exponential':
                descr.get_exponential_tiles(distribution=distribution, max_tiles=max_tiles)
            else:
                print("Unsupported operation model", args.exploration_model)
                assert False

            cgras.append(descr)
    elif args.manual_distribution:
        rows, cols, tiles, distribution, max_tiles = load_cgra_tiles_from_manual_distribution(args.CGRA)

        random.seed(1) # I don't thik this does random stuff, but justs to be sure.
        cgras = [CGRADescription(rows, cols, tiles=tiles)]
        
    else:
        print ("Unsupported mode --- use --random-cgras and pass exploration mode")
        assert False
        pass

    if args.print_cgras:
        for i in range(len(cgras)):
            print("CGRA " + str(i) + ":", str(cgras[i]))

    # TODO -- pre-compile this stuff to save time.
    benchmarks = load_benchmarks_from_json(args, args.benchmark_files)
    # benchmarks = [Benchmark(file, args, prog_features=args.with_program_features) for file in args.benchmarks]
    ModelCatalog.register_custom_model(
        "placement_model", PlacementModel
    )

    env = PlacementEnv(cgras, benchmarks, rows, cols, args, cgra_state=not args.no_cgra_state)
    class RaylibEnv(PlacementEnv):
        def __init__(self, env_config):
            print("Creating env with config" + str(env_config))
            super().__init__(cgras, benchmarks, rows, cols, args, cgra_state=not args.no_cgra_state)

    if args.annealer:
        print("Running annealer")
        random.seed(0)
        j = 0
        for cgra_desc in cgras:
            cgra = CGRA(benchmarks, rows, cols, args)
            tiles = cgra_desc.tiles[:]
            random.shuffle(tiles)
            for i in range(len(tiles)):
                cgra.add_operation_to(i % (cols * rows), tiles[i])
            annealer = Annelaer(cgra_desc)
            result = annealer.anneal(cgra)
            print("For CGRA", j)
            j += 1
            print("Annealing result is ", result)
            print("Steps required was ", annealer.tracker['Steps'])
    elif args.test:
        print("Testing RL")
        from ray.rllib.agents.ppo import PPOTrainer
        config = {
            "model": {
                "custom_model": "placement_model",
                "custom_model_config": {
                    "rows": rows,
                    "cols": cols,
                    "true_obs_space": None,
                    "with_features": args.with_program_features,
                    "no_cgra_state": args.no_cgra_state,
                },
            },
            "evaluation_duration": len(cgras),
            "evaluation_duration_unit": "episodes",
            "framework": "torch",
            "gamma": 1.0,
            # Tweak the default model provided automatically by RLlib,
            # given the environment's observation- and action spaces.
            # "model": {
            #     "fcnet_hiddens": [64, 64],
            #     "fcnet_activation": "relu",
            #     "conv_filters": [[64, [3, 3], 2]],
            # },
            # Set up a separate evaluation worker set for the
            # `trainer.evaluate()` call after training (see below).
            "evaluation_num_workers": 50,
            "evaluation_config": {
                "explore": False
            }
        }

        # Create our RLlib Trainer.
        env_config = {
            "programs": benchmarks,
            "rows": rows,
            "cols": cols,
            "tileopts": tiles,
            "args": args,
            "mask": True
        }

        tune.register_env('PlacementEnv', lambda env_name: RaylibEnv(env_config=config))
        import ray
        ray.rllib.utils.check_env(env)

        trainer = PPOTrainer(env='PlacementEnv', config=config)
        if args.restore is None:
            print("If setting --test, must also set --restore")
            sys.exit(1)
        else:
            trainer.restore(args.restore)
            print("Restoring old model found in", args.restore)
        print("Running evaluation")
        trainer.evaluate()
        print("Done with evalation")
    else:
        print("Running RL")
        # Import the RL algorithm (Trainer) we would like to use.
        from ray.rllib.agents.ppo import PPOTrainer

        # Configure the algorithm.
        config = {
            "model": {
                "custom_model": "placement_model",
                "custom_model_config": {
                    "rows": rows,
                    "cols": cols,
                    "true_obs_space": None,
                    "with_features": args.with_program_features,
                    "no_cgra_state": args.no_cgra_state,
                },
            },
            # Use 2 environment workers (aka "rollout workers") that parallelly
            # collect samples from their own environment clone(s).
            "num_workers": 1,
            # Change this to "framework: torch", if you are using PyTorch.
            # Also, use "framework: tf2" for tf2.x eager execution.
            "framework": "torch",
            "gamma": 1.0,
            "lr": 0.001,
            # Tweak the default model provided automatically by RLlib,
            # given the environment's observation- and action spaces.
            # "model": {
            #     "fcnet_hiddens": [64, 64],
            #     "fcnet_activation": "relu",
            #     "conv_filters": [[64, [3, 3], 2]],
            # },
            # Set up a separate evaluation worker set for the
            # `trainer.evaluate()` call after training (see below).
            "evaluation_num_workers": 10,
            "evaluation_config": {
                "explore": False
            }
            # Only for evaluation runs, render the env.
        }

        # Create our RLlib Trainer.
        env_config = {
            "programs": benchmarks,
            "rows": rows,
            "cols": cols,
            "tileopts": tiles,
            "args": args,
            "mask": True
        }

        tune.register_env('PlacementEnv', lambda env_name: RaylibEnv(env_config=config))
        import ray
        ray.rllib.utils.check_env(env)

        trainer = PPOTrainer(env='PlacementEnv', config=config)
        if args.restore is not None:
            trainer.restore(args.restore)
            print("Restoring old model found in", args.restore)

        improving = True
        last_10_episode_rewards = []
        i = 0
        while improving:
            print("Starting iteration ", i)
            result = trainer.train()
            print ("Episode reward mean is ", result['episode_reward_mean'])
            print ("Episode rewards is ", result.keys())
            i += 1
            last_10_episode_rewards.append(result['episode_reward_mean'])
            if len(last_10_episode_rewards) > 10:
                last_5_mean = np.mean(np.array(last_10_episode_rewards[5:]))
                first_5_mean = np.mean(np.array(last_10_episode_rewards[:5]))
                last_10_episode_rewards = last_10_episode_rewards[1:] # Drop the oldest element.
                print("First 5 mean is ", first_5_mean)
                print("Last 5 mean is ", last_5_mean)

                if first_5_mean < last_5_mean:
                    improving = False

            print("iteration done")
            if i % 100 == 0:
                checkpoint = trainer.save()
                print("Saved checkpoint ", checkpoint)
            if i % 10 == 0:
                # Do an evaluation on unseen models.
                print("Running evaluation")
                trainer.evaluate()
                print("Evaluation done")

        # Final save:
        if args.save_model:
            checkpoint = trainer.save()
            os.rename(checkpoint, args.save_model)
            os.rename(checkpoint + '.tune_metadata', args.save_model + '.tune_metadata')
            print("Saved model as ", args.save_model)