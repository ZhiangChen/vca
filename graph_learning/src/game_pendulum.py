#!/usr/bin/env python2

"""
game_pendulum.py
Zhiang Chen
4/19/2018
"""

"""Agent"""
from brain import RLAgent

"""Environment"""
import gym
from gym import wrappers

"""Auxiliary"""
from rl.normalized_actions import NormalizedActions
from rl.ounoise import OUNoise
from rl.replay_memory import ReplayMemory, Transition
import argparse
import torch
import numpy as np

class Batch(object):
    def __init__(self, batch):
        E = torch.cat([s[0] for s in batch.state], 0)
        S = torch.cat([s[1] for s in batch.state], 0)
        G = torch.cat([s[2] for s in batch.state], 0)
        E = E.unsqueeze(1)
        self.state = (E,S,G)

        E = torch.cat(batch.next_state, 0)
        E = E.unsqueeze(1)
        self.next_state = (E,S,G)

        self.action = torch.cat(batch.action, 0)

        self.reward = torch.cat(batch.reward, 0).unsqueeze(1)

        self.mask = torch.cat(batch.mask, 0).unsqueeze(1)


class penDulum(object):
    def __init__(self, gamma, tau, replay_size):
        env_name = 'Pendulum-v0'
        env = NormalizedActions(gym.make(env_name))
        self.env = wrappers.Monitor(env, '/tmp/{}-experiment'.format(env_name), force=True)
        self.env.seed(0)

        self.memory = ReplayMemory(replay_size)
        self.ounoise = OUNoise(self.env.action_space.shape[0])


        fc_param = (self.env.observation_space.shape[0], 16, 32)
        rl_param = (128, 12, self.env.action_space.shape[0])
        self.agent = RLAgent(gamma, tau, fc_param, rl_param)


    def train(self, n_eps):
        rewards = []
        for i_episode in range(n_eps):
            if i_episode < n_eps // 2:
                e = torch.Tensor([self.env.reset()])
                self.ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                                  i_episode) / args.exploration_end + args.final_noise_scale
                self.ounoise.reset()
                episode_reward = 0
                for t in range(args.num_steps):
                    s = torch.zeros(1,1)
                    g = torch.zeros(1,1)
                    action = self.agent.select_action(e, s, g, self.ounoise)
                    next_state, reward, done, _ = self.env.step(action.numpy()[0])
                    episode_reward += reward

                    action = torch.Tensor(action)
                    mask = torch.Tensor([not done])
                    next_state = torch.Tensor([next_state])
                    reward = torch.Tensor([reward])

                    if i_episode % 10 == 0:
                        self.env.render()

                    self.memory.push((e, s, g), action, mask, next_state, reward)

                    e = next_state

                    if len(self.memory) > args.batch_size * 5:
                        for _ in range(args.updates_per_step):
                            transitions = self.memory.sample(args.batch_size)
                            batch = Transition(*zip(*transitions))

                            b = Batch(batch)
                            self.agent.update_parameters(b)

                    if done:
                        break
                rewards.append(episode_reward)

            else:
                e = torch.Tensor([self.env.reset()])
                episode_reward = 0
                for t in range(args.num_steps):
                    s = torch.zeros(1, 1)
                    g = torch.zeros(1, 1)
                    action = self.agent.select_action(e, s, g)

                    next_state, reward, done, _ = env.step(action.numpy()[0])
                    episode_reward += reward

                    next_state = torch.Tensor([next_state])

                    if i_episode % 10 == 0:
                        self.env.render()

                    e = next_state
                    if done:
                        break

                rewards.append(episode_reward)
            print("Episode: {}, noise: {}, reward: {}, average reward: {}".format(i_episode, self.ounoise.scale, rewards[-1], np.mean(rewards[-100:])))

""" Hyperparameters """
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--render', action='store_true', help='render the environment')

args = parser.parse_args()


if __name__ == '__main__':
    torch.manual_seed(0)
    game = penDulum(args.gamma, args.tau, args.replay_size)
    game.train(500)