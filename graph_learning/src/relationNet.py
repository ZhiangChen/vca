#!/usr/bin/env python2

"""
relationNet.py
Zhiang Chen
4/14/2018
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F


from fc import *
from resnet import *
from rl.ddpg import DDPG
from rl.naf import NAF
from rl.normalized_actions import NormalizedActions
from rl.ounoise import OUNoise
from rl.replay_memory import ReplayMemory, Transition

class relationNet(nn.Module):
    pass




