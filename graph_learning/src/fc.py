#!/usr/bin/env python2

"""
fc.py
Zhiang Chen
4/11/2018
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FC2Layers(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, activation=F.relu):
        super(FC2Layers, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class FC2LayersShortcut(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, activation=F.relu):
        super(FC2LayersShortcut, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden+n_in, n_out)
        self.activation = activation

    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = torch.cat((h,x),1)
        x = self.fc2(h)
        return x


"""
net = Net()
i = Variable(torch.randn(3,2))
print(net(i))
#"""
