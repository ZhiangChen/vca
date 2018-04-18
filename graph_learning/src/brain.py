#!/usr/bin/env python2

"""
brain.py
Zhiang Chen
4/18/2018
"""
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

from relationaNet import relationNet
from rl.ddpg import DDPG
from rl.naf import Policy
from rl.normalized_actions import NormalizedActions
from rl.ounoise import OUNoise
from rl.replay_memory import ReplayMemory, Transition

"""PAPAMETERS TO TUNE"""
# input (relation effect, state, goal), dim = 10 + 5 + 2
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')

# FC2LayersShortcut architecture
# ResNet architecture
# activation function of MLPs output
# input of relationNet
# input of NAF (relation effect, state, goal), dim = 10 + 5 + 2
n_in = 12
n_hidden = 16
n_out = 32
fc_param = (n_in, n_hidden, n_out)


class RLAgent(nn.Module):
    def __init__(self, gamma, tau, fc_param):
        self.action_space = action_space
        self.num_inputs = num_inputs

        self.model = relationNet(*fc_param)
        self.target_model = relationNet(*fc_param)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

    def select_action(self, state, exploration=None):
        pass



