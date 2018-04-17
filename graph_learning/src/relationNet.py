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


# state of a robot (x, y, alpha, v_x, v_y)
# one-hot coding relation (1,0), (0,1)

"""PAPAMETERS TO TUNE"""
# FC2LayersShortcut architecture
# ResNet architecture
# activation function of MLPs output
# input of relationNet

class relationNet(nn.Module):
    def __init__(self):
        super(relationNet, self).__init__()
        self.mlp = FC2LayersShortcut(12, 16, 32)
        self.res = ResNet18()

    def forward(self, x):
        """
        :param x: Variable, [num_edge, edge_dim]
        :return:
        """
        h = self.mlp(x)
        h = F.selu(h)
        out_prd = torch.bmm(h.unsqueeze(2), h.unsqueeze(1)) # outer product
        r_map = torch.sum(out_prd, -3) # relation map
        relation = self.res(r_map.view(-1, 1, 32, 32))
        return relation

if __name__ == '__main__':
    torch.manual_seed(0)
    net = relationNet()
    i = Variable(torch.randn(4, 12))
    print net(i)