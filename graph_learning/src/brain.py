#!/usr/bin/env python2

"""
brain.py
Zhiang Chen
4/17/2018
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

from relationaNet import relationNet
from rl.ddpg import DDPG
from rl.naf import NAF
from rl.normalized_actions import NormalizedActions
from rl.ounoise import OUNoise
from rl.replay_memory import ReplayMemory, Transition

class RLAgent(nn.Module):
    def __init__(self):
        super(RLAgent, self).__init__()
        self.r_net = relationNet()

