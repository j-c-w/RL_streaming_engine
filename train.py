import torch
import argparse
import numpy as np
import networkx as nx
from collections import deque
from matplotlib import pyplot as plt
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import dgl
from net import PolicyNet
from env import StreamingEngineEnv
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
from util import calc_score, initial_fill, ROW, COL, fix_grid_bins
from torch.utils.tensorboard import SummaryWriter
import time

#torch.autograd.set_detect_anomaly(True)
# random.seed(10)

def get_args():
    parser = argparse.ArgumentParser(description='grid placement')
    arg = parser.add_argument
    arg('--mode', type=int, default=1, help='0 random search, 1 CMA-ES search, 2- RL PPO, 3- sinkhorn')

    arg('--grid_size',   type=int, default=4, help='number of sqrt PE')
    arg('--spokes',   type=int, default=3, help='Number of spokes')
    arg('--epochs',   type=int, default=10000, help='number of iterations')
    arg('--nodes', type=int, default=20,  help='number of nodes')
    arg('--debug', dest='debug', action='store_true', default=False, help='debug mode')

    # PPO
    arg('--num-episode', type=int, default=100000)
    arg('--ppo-epoch', type=int, default=4)
    arg('--max-grad-norm', type=float, default=1)
    arg('--graph_size', type=int, default=128, help='graph embedding size')
    arg('--emb_size', type=int, default=128, help='embedding size')
    arg('--update_timestep', type=int, default=500, help='update policy every n timesteps')
    arg('--K_epochs', type=int, default=100, help='update policy for K epochs')
    arg('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    arg('--gamma', type=float, default=0.99, help='discount factor')
    arg('--lr', type=float, default=1e-3, help='parameters for Adam optimizer')
    arg('--betas', type=float, default=(0.9, 0.999), help='')
    arg('--loss_entropy_c', type=float, default=0.01, help='coefficient for entropy term in loss')
    arg('--loss_value_c', type=float, default=0.5, help='coefficient for value term in loss')
    arg('--model', type=str, default='', help='load saved model')
    arg('--log_interval', type=int, default=100, help='interval for log')

    args = parser.parse_args()
    return args

# Specify graph as (src_ids, dst_ids) edges, node ordering starts from 0 regardless of specification
PREDEF_GRAPHS = {
    "DISTANCE": ([0, 1, 2, 2, 2, 3, 4, 4, 5, 5, 6, 7, 8, 8, 11, 12, 12, 13, 13, 14, 14, 14, 15, 16, 17, 18, 19, 19, 20, 20, 21, 22],
                 [1, 2, 3, 4, 5, 4, 5, 6, 6, 7, 7, 8, 9, 10, 12, 13, 19, 15, 14, 16, 17, 18, 19, 17, 18, 19, 20, 21, 21, 22, 22, 23]), 
    "FFT_OLD":
             ([0,
               2, 3,
               5, 6, 6, 7, 8, 9, 10, 11, 12,
               14, 15, 16, 17, 18, 19, 20, 21, 17, 17,
               23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
               35, 35, 36, 37, 38, 38, 39, 39, 40, 42, 41, 43, 43, 48, 49, 50, 51,
               53,
               55,
              ],
              [1,
               3, 4,
               6, 7, 8, 9, 9, 10, 11, 12, 13,
               15, 16, 17, 18, 19, 20, 21, 22, 19, 21,
               24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
               36, 37, 38, 39, 40, 42, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
               54,
               56,
              ]),
    "FFT_SIMPLE": ([0, 0, 1, 1, 2, 3, 4, 4, 5], [1, 2, 3, 4, 4, 6, 5, 8, 7]),
    "FFT_SYNC2": ([1, 1, 2, 2, 3, 4, 6, 6, 8], [2, 3, 4, 6, 6, 5, 7, 8, 9])
}


if __name__ == "__main__":
    args = get_args()  # Holds all the input arguments
    print('Arguments:', args)
    writer = SummaryWriter()

    graphdef = PREDEF_GRAPHS["FFT_SYNC2"]
    # random generate a directed acyclic graph
    if graphdef is None:
        a = nx.generators.directed.gn_graph(args.nodes)
        graph = dgl.from_networkx(a)
    else:
        graph = dgl.graph((torch.Tensor(graphdef[0]).int(), torch.Tensor(graphdef[1]).int()))

    args.nodes = nodes = graph.number_of_nodes()

    if args.debug:
        graph_in = graph.adjacency_matrix_scipy().toarray()
        print('graph adjacency matrix: ', graph_in)
        nx_g = graph.to_networkx()
        nx.draw(nx_g, nx.nx_agraph.graphviz_layout(nx_g, prog='dot'), with_labels=True)
        plt.show()

    # random search
    if args.mode == 0:
        # randomly occupy with nodes (not occupied=0 value):
        device_topology = (16, 1, args.spokes)
        # device_topology = (args.grid_size, args.grid_size, args.spokes)
        grid, grid_in, place = initial_fill(nodes, device_topology)

        env = StreamingEngineEnv(compute_graph_def=graphdef,
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=48,
                                 graph_feat_size=32)

        # testing grid placement scoring:
        _, ready_time, valid = env._calculate_reward(torch.tensor(grid_in))

        # random search
        before_rs = ready_time.max().item() if valid else float('inf')
        best_grid = grid_in.copy()

        print('Running Random search optimization ...')
        for i in tqdm(range(args.num_episode)):
            grid, grid_in, _ = initial_fill(nodes, grid.shape)
            _, ready_time, valid = env._calculate_reward(torch.tensor(grid_in))
            after_rs = ready_time.max().item()
            if before_rs > after_rs and valid:
                before_rs = after_rs
                best_grid = grid_in.copy()
                if args.debug:
                    print('best_grid score so far: ', after_rs)

        print('best score found: ', before_rs)
        if args.debug:
            print('optim placement: ', best_grid)

    # ES search
    if args.mode == 1:
        # randomly occupy with nodes (not occupied=0 value):
        device_topology = (16, 1, args.spokes)
        # device_topology = (args.grid_size, args.grid_size, args.spokes)
        grid, grid_in, place = initial_fill(nodes, device_topology)

        env = StreamingEngineEnv(compute_graph_def=graphdef,
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=48,
                                 graph_feat_size=32)

        # testing grid placement scoring:
        _, ready_time, valid = env._calculate_reward(torch.tensor(grid_in))
        final_es = ready_time.max().item() if valid else float('inf')
        final_value = grid_in if valid else None

        import nevergrad as ng

        budget = args.num_episode  # How many steps of training we will do before concluding.
        workers = 16
        # param = ng.p.Array(shape=(int(nodes), 1)).set_integer_casting().set_bounds(lower=0, upper=ROW*COL*nodes)
        param = ng.p.Array(init=place).set_integer_casting().set_bounds(lower=0, upper= np.prod(device_topology))
        # ES optim
        names = "CMA"
        optim = ng.optimizers.registry[names](parametrization=param, budget=budget, num_workers=workers)
        # optim = ng.optimizers.RandomSearch(parametrization=param, budget=budget, num_workers=workers)
        # optim = ng.optimizers.NGOpt(parametrization=param, budget=budget, num_workers=workers)

        print('Running ES optimization ...')
        for _ in tqdm(range(budget)):
            x = optim.ask()
            grid, grid_in, _ = initial_fill(nodes, device_topology, manual=x.value)
            _, ready_time, valid = env._calculate_reward(torch.tensor(grid_in))
            loss = ready_time.max().item() if valid else float('inf')
            optim.tell(x, loss)
            if final_es > loss:
                final_value = grid_in
                final_es = loss

        rec = optim.recommend()
        grid, grid_in, _ = initial_fill(nodes, device_topology, manual=rec.value)
        _, ready_time, valid = env._calculate_reward(torch.tensor(grid_in))
        
        print('best score found:', final_es)
        if args.debug:
            print('optim placement:\n', final_value)

    # PPO
    if args.mode == 2:
        from ppo_discrete import PPO

        device_topology = (16, 1, args.spokes)
        # RL RNN place each node
        env = StreamingEngineEnv(compute_graph_def=graphdef,
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=48,
                                 graph_feat_size=32)
        # env = GridEnv(args, grid_in, graph)
        ppo = PPO(args, env)

        # logging variables
        running_reward = 0
        time_step = 0
        avg_improve = 0

        start = time.time()
        # training loop:
        print('Starting PPO training...')
        for i_episode in range(1, args.epochs + 1):
            state, initial_rl = env.reset()
            gr_edges = torch.stack(env.graph.edges()).unsqueeze(0).float()
            best_reward = initial_rl
            for node in range(args.nodes):
                time_step += 1
                node_1hot = torch.zeros(args.nodes)
                node_1hot[node] = 1.0
                rl_state = torch.cat((torch.FloatTensor(state).view(-1), node_1hot))  # grid, node to place
                action = ppo.select_action(rl_state, gr_edges)
                state, reward = env.step(action, node)
                # print('node: ', node); print('action: ', action); print('reward: ', reward); print('state: ', state)
                # input()
                # Saving reward and is_terminals:
                ppo.buffer.rewards.append(reward)
                if node == (args.nodes - 1):
                    done = True
                else:
                    done = False
                ppo.buffer.is_terminals.append(done)

                # learning:
                if time_step % args.update_timestep == 0:
                    ppo.update()
                    time_step = 0
                running_reward += reward
                best_reward = max(best_reward, reward)

            # print(final_rl, best_reward, (abs(final_rl) - abs(best_reward)))
            avg_improve += (abs(initial_rl) - abs(best_reward))

            if initial_rl > best_reward:
                print('got worse: ', initial_rl, best_reward)

            # logging
            if i_episode % args.log_interval == 0:
                running_reward = int((running_reward / args.log_interval))
                avg_improve = int(avg_improve / args.log_interval)
                writer.add_scalar('running_reward/episode', running_reward, i_episode)
                print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))
                writer.add_scalar('final_reward/episode', best_reward, i_episode)
                print('Episode {} \t Final reward: {}'.format(i_episode, best_reward))
                end = time.time()
                print('Execution time {} s'.format(end - start))
                # writer.add_scalar('avg improvement/episode', avg_improve, i_episode)
                # print('Episode {} \t Avg improvement: {}'.format(i_episode, avg_improve))
                torch.save(ppo.policy.state_dict(), 'model_epoch_' + str(avg_improve) + '.pth')
                running_reward = avg_improve = 0

    # sinkhorn
    if args.mode == 3:
        device_topology = (16, 1, args.spokes)
        # device_topology = (args.grid_size, args.grid_size, args.spokes)
        grid, grid_in, place = initial_fill(nodes, device_topology)

        # initialize Environment, Network and Optimizer
        env = StreamingEngineEnv(compute_graph_def=graphdef,
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=48, 
                                 graph_feat_size=32,
                                 init_place=torch.tensor(grid_in),
                                 emb_mode='init')
        policy = PolicyNet(cg_in_feats=32, 
                           cg_hidden_dim=64, 
                           cg_conv_k=1, 
                           transformer_dim=48, 
                           transformer_nhead=4, 
                           transformer_ffdim=128, 
                           transformer_dropout=0.1, 
                           transformer_num_layers=4, 
                           sinkhorn_iters=100)
        optim = Adam(policy.parameters(), lr=args.lr)

        # to keep track of average reward
        reward_buf = deque(maxlen=100)
        reward_buf.append(0)

        # train
        for episode in range(args.num_episode):

            # reset env
            state = env.reset()

            # collect 'trajectory'
            # only one trajectory step
            with torch.no_grad():
                old_action, old_logp, old_entropy, old_scores = policy(*state)
                old_reward, _, _ = env.step(old_action)

            # use 'trajectory' to train network
            for epoch in range(args.ppo_epoch):

                action, logp, entropy, scores = policy(*state)
                reward, ready_time, _ = env.step(action)

                ratio = torch.exp(logp) / (torch.exp(old_logp) + 1e-8)
                surr1 = ratio * reward
                surr2 = torch.clamp(ratio, 1-args.eps_clip, 1+args.eps_clip) * reward
                action_loss = -torch.fmin(surr1, surr2)
                entropy_loss = entropy * args.loss_entropy_c

                loss = (action_loss + entropy)
                loss = loss.mean()

                optim.zero_grad()
                loss.backward()
                clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optim.step()

                reward_buf.append(reward.mean())
            if episode % 100 == 0:
                print(f'Episode: {episode} | Epoch: {epoch} | Ready time: {ready_time.max().item()} | Loss: {loss.item()} | Mean Reward: {np.mean(reward_buf)}')
                # TODO: Is mean of reward buffer the total mean up until now?
                writer.add_scalar('loss/episode', loss.item(), episode)
                writer.add_scalar('total time/episode', ready_time.max().item(), episode)
                writer.flush()

            if episode % 100 == 0 and args.debug:
                plt.imshow(old_scores[0].exp().detach(), vmin=0, vmax=1)
                plt.pause(1e-6)
                #plt.show()
                #env.render()



