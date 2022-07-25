import os
import pickle
from queue import Queue
import time
import argparse
import logging
from collections import deque
from modules import RolloutBuffer

import dgl
import torch
from util import get_graph_json, create_graph, output_json, print_graph
from preproc import PreInput
import numpy as np
from tqdm import tqdm
import random
from multiprocessing import Pool
import multiprocessing as mp

from envs.streaming_engine_env import StreamingEngineEnv
from ppo_discrete import PPO
from modules import ActorCritic

import train

_engine = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Classify what the required improvement rate should be.
class AnnealerPPO:
    def __init__(
        self,
        args,
        graphdef,
        device,
        state_dim
    ):
        self.args = args
        self.device_topology = device['topology']
        self.gnn_in = graphdef['graph'].ndata['feat'].shape[1]
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(args=args,
            device=device,
            state_dim=state_dim,
            emb_size=args.emb_size,
            action_dim=2,
            graph_feat_size=self.args.graph_feat_size,
            gnn_in=self.gnn_in,
            node_ids=False,
            raw_values=True
        )
        self.policy_old = self.policy
        if args.model != '':
            self.load(args.model)

        self.MseLoss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr, betas=self.args.betas)

    def select_action(self, tensor_in, graphdef):
        with torch.no_grad():
            graph_info = graphdef['graph'].to(_engine)
            state = torch.FloatTensor(tensor_in).to(_engine)
            action, action_logprob = self.policy_old.act(state, graph_info, None, None)

        return action, (state, action, graph_info, action_logprob)

    def add_buffer(self, inbuff, reward, done):
        (state, action, graphinfo, action_logprob) = inbuff
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.graphs.append(graphinfo)
        self.buffer.logits.append(action_logprob)
        self.buffer.is_terminals.append(done)
        self.buffer.rewards.append(reward)

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            discounted_reward = reward + (self.args.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            if is_terminal:
                discounted_reward = 0
        rewards = torch.tensor(rewards, dtype=torch.float32).to(_engine)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        else:
            rewards = rewards - rewards.mean()

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(_engine)
        old_graph = [graph.to(_engine) for graph in self.buffer.graphs]
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(_engine)
        old_logits = torch.squeeze(torch.stack(self.buffer.logits, dim=0)).detach().to(_engine)

        for _ in range(self.args.K_epochs):
            logits, state_values = self.policy.evaluate(old_states, old_actions, old_graph, None, None)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logits - old_logits.detach()).mean(1)

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + \
                    + self.args.loss_value_c*self.MseLoss(state_values, rewards) + \
                    - self.args.loss_entropy_c

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        print("Loading model")
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

# Run an RL algorithm over the paramters of the simulated annealer.
if __name__ == "__main__":
    pass