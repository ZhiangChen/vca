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
from rl.naf import Policy
from rl.normalized_actions import NormalizedActions
from rl.ounoise import OUNoise
from rl.replay_memory import ReplayMemory, Transition


# state of a robot (x, y, alpha, v_x, v_y)
# one-hot coding relation (1,0), (0,1)

"""PARAMETERS TO TUNE"""
# FC2LayersShortcut architecture
# ResNet architecture
# activation function of MLPs output
# input of relationNet
# input of NAF (relation effect, state, goal), dim = 10 + 5 + 2
n_in = 10
n_hidden = 16
n_out = 32
fc_param = (n_in, n_hidden, n_out)

hidden_size = 128
num_inputs = 17
action_space = 2
rl_param = (hidden_size, num_inputs, action_space)

class relationNet(nn.Module):
    def __init__(self, fc_param, rl_param = (128, 17, 2)):  # constructor parameter is a list
        super(relationNet, self).__init__()
        self.fc_param = fc_param
        self.mlp = FC2LayersShortcut(*fc_param)
        self.res = ResNet18()
        self.naf = Policy(*rl_param)

    def forward(self, (e, s, g), u):
        """
        :param e: edge, shape=(batch_num, edge_num, edg_dim)
        :param s: state of a robot (x, y, alpha, v_x, v_y)
        :param g: goal (x, y)
        :param u: velocity control (v_x, v_y)
        :return: mu, Q, V, see naf.py
        """
        n = e.size()[-2]  # number of edges received by one robot
        e = e.view(-1,self.fc_param[0])
        h = self.mlp(e)
        h = F.selu(h)
        out_prd = torch.bmm(h.unsqueeze(2), h.unsqueeze(1))  # outer product
        out_prd = out_prd.view(-1, n, 32, 32)  # reshape back to batch
        r_map = torch.sum(out_prd, -3)  # relation map
        relation = self.res(r_map.view(-1, 1, 32, 32))
        i = torch.cat((relation, s, g), 1)
        return self.naf((i,u))


if __name__ == '__main__':
    torch.manual_seed(0)
    net = relationNet(fc_param, rl_param)
    net.eval()
    e = Variable(torch.randn(4, 2, n_in), volatile=True)
    s = Variable(torch.randn(4, 5), volatile=True)
    g = Variable(torch.randn(4, 2), volatile=True)
    u = Variable(torch.randn(4, 2), volatile=True)
    print net((e, s, g), u)
    net.train()
    e = Variable(torch.randn(3, 4, n_in), volatile=True)
    s = Variable(torch.randn(3, 5), volatile=True)
    g = Variable(torch.randn(3, 2), volatile=True)
    u = Variable(torch.randn(3, 2), volatile=True)
    print net((e, s, g), None)