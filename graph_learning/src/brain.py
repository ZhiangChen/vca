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

from relationNet import relationNet
from rl.ddpg import DDPG
from rl.naf import Policy
from rl.normalized_actions import NormalizedActions
from rl.ounoise import OUNoise
from rl.replay_memory import ReplayMemory, Transition

"""GLOBAL ARGUMENT"""
MSELoss = nn.MSELoss()
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

""" PARAMETERS TO TUNE - all from relationNet.py """
# FC2LayersShortcut architecture
# ResNet architecture
# activation function of MLPs output
# input of relationNet
# input of NAF (relation effect, state, goal), dim = 10 + 5 + 2
n_in = 12
n_hidden = 16
n_out = 32
fc_param = (n_in, n_hidden, n_out)


class RLAgent(object):
    def __init__(self, gamma, tau, fc_param):
        """ relationNet returns mu, Q, V """
        self.model = relationNet(fc_param)
        self.target_model = relationNet(fc_param)

        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.gamma = gamma
        self.tau = tau

    def select_action(self, e, s, g, exploration=None):
        self.model.eval()
        e = Variable(e, volatile=True)
        s = Variable(s, volatile=True)
        g = Variable(g, volatile=True)
        mu, _, _ = self.model((e, s, g), u=None)
        self.model.train()
        mu = mu.data
        if exploration is not None:
            mu += torch.Tensor(exploration.noise())

        return mu.clamp(-1, 1)

    def update_parameters(self, batch):
        state_batch = tuple([Variable(s) for s in batch.state])
        next_state_batch = tuple([Variable(s, volatile=True) for s in batch.next_state])
        action_batch = Variable(batch.action)
        reward_batch = Variable(batch.reward)

        _, _, next_state_values = self.target_model(next_state_batch, None)

        expected_state_action_values = reward_batch + (next_state_values * self.gamma)

        _, state_action_values, _ = self.model(state_batch, action_batch)

        loss = MSELoss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
        self.optimizer.step()

        soft_update(self.target_model, self.model, self.tau)

    def save_model(self, file="model", path="./trained_models/", save_target=False):
        torch.save(self.model.state_dict(), path+file)
        if save_target:
            torch.save(self.target_model.state_dict(), path+file+"_target")

    def load_model(self, file="model", path="./trained_models/", load_target=False):
        self.model.load_state_dict(torch.load(path+file))
        if load_target:
            self.target_model.load_state_dict(torch.load(path+file+"_target"))


""" Parameters for testing """
parser = argparse.ArgumentParser(description='RelationNet Hyperparameters')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
args = parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(0)
    agent = RLAgent(args.gamma, args.tau, fc_param)
    e = torch.randn(4,12)
    s = torch.randn(1,5)
    g = torch.randn(1,2)
    print "select action: "
    print agent.select_action(e, s, g)

    class batch:
        def __init__(self):
            e = torch.randn(2, 4, 12)
            s = torch.randn(2, 5)
            g = torch.randn(2, 2)
            a = torch.randn(2, 2)
            r = torch.randn(2, 1)
            self.state = (e,s,g)
            self.next_state = (e,s,g)
            self.action = a
            self.reward = r

    data = batch()
    agent.update_parameters(data)

    agent.save_model()

    agent2 = RLAgent(args.gamma, args.tau, fc_param)
    agent2.load_model()






