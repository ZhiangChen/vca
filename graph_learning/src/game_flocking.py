#!/usr/bin/env python2

"""
game_flocking.py
Zhiang Chen
4/21/2018
"""

"""Agent"""
from brain import RLAgent

"""Environment"""
from env import Env
import rospy

"""Auxiliary"""
from rl.normalized_actions import NormalizedActions
from rl.ounoise import OUNoise
from rl.replay_memory import ReplayMemory, Transition
from batch import Memory
import argparse
import torch
import numpy as np
import time
from std_msgs.msg import Float32

""" Hyperparameters """
# FC2LayersShortcut architecture
# ResNet architecture
# activation function of MLPs output
# input of relationNet
# input of NAF (relation effect, state, goal), dim = 10 + 6 + 2
# gamma from Q learning
# tau from soft parameter updating

gamma = 0.99
tau = 0.001
fc_param = (14, 16, 32)
rl_param = (128, 18, 2)

exploration_end = 100
final_noise_scale = 0.3
noise_scale = 0.3

batch_size = 64
updates_per_eps = 10
n_steps = 200
replay_size = 1000000

class Game(object):
    def __init__(self, env_id=0, n_robot=2, dist=1.5):
        """
        :param env_id: 0 - two robots one group; 1 - four robots two group
        :param n_robot: 2; 4
        :param dist: 1.5
        """
        self.pub = rospy.Publisher('reward', Float32, queue_size=1)
        self.env = Env(env_id, n_robot, dist)

        self.memory = Memory(replay_size)
        self.ounoise = OUNoise(2)

        self.agent = RLAgent(gamma, tau, fc_param, rl_param)

        """
        self.memory.load()
        self.agent.load_model(load_target=True)
        self.train_0(100)
        self.agent.save_model(save_target=True)
        self.train_0(50)
        self.agent.save_model(save_target=True)
        """
        self.train_1(2)

    def train_0(self, n_eps):
        rewards = []
        relation = torch.Tensor((0, 1))


        for i_episode in range(n_eps):
            robot_state = self.env.getState()
            s0 = torch.Tensor(robot_state[0][0] + robot_state[0][1])/10.0
            s1 = torch.Tensor(robot_state[1][0] + robot_state[1][1])/10.0
            e0 = torch.cat((s1, s0, relation)).unsqueeze(0).unsqueeze(0)
            e1 = torch.cat((s0, s1, relation)).unsqueeze(0).unsqueeze(0)
            s0 = s0.unsqueeze(0)
            s1 = s1.unsqueeze(0)
            goal = torch.Tensor(self.env.generateGoal()).unsqueeze(0)/10.0
            goals = tuple([tuple(goal.numpy()[0]) for _ in range(2)])


            print "goal: " + str(goal.numpy())

            self.ounoise.scale = (noise_scale - final_noise_scale) * max(0, exploration_end - i_episode) / exploration_end + final_noise_scale
            self.ounoise.reset()
            episode_reward = 0
            n_batch = updates_per_eps


            for i_step in range(n_steps):
                a0 = self.agent.select_action(e0, s0, goal, self.ounoise)
                a1 = self.agent.select_action(e1, s1, goal, self.ounoise)
                vel = tuple([tuple(a) for a in torch.cat((a0, a1), 0).numpy()])
                robot_state_, reward, done = self.env.step(vel, goals)
                episode_reward += reward

                esga0 = (e0, s0, goal, a0)
                esga1 = (e1, s1, goal, a1)

                robot_state = robot_state_
                s0 = torch.Tensor(robot_state[0][0] + robot_state[0][1])/10.0
                s1 = torch.Tensor(robot_state[1][0] + robot_state[1][1])/10.0
                e0 = torch.cat((s1, s0, relation)).unsqueeze(0).unsqueeze(0)
                e1 = torch.cat((s0, s1, relation)).unsqueeze(0).unsqueeze(0)
                s0 = s0.unsqueeze(0)
                s1 = s1.unsqueeze(0)

                es0_ = (e0, s0)
                es1_ = (e1, s1)

                reward = reward.tolist()
                r0 = reward[0]
                r1 = reward[1]

                self.pub.publish(r0)

                self.memory.push((esga0, es0_, r0))
                self.memory.push((esga1, es1_, r1))

                if done:
                    break

            self.memory.save()

            if len(self.memory) > batch_size * n_batch:
                n_batch += 6
                print "learning..."
                for _ in range(updates_per_eps):
                    batch = self.memory.sample(batch_size)
                    self.agent.update_parameters(batch)

            rewards.append(episode_reward)
            print("Episode: {}, noise: {}, return: {}, "
                  "average return: {}".format(i_episode, self.ounoise.scale,
                                              rewards[-1], np.mean(rewards[0][-100:])))

            print "random walking"
            self.env.reset()
            print '\n'

    def train_1(self, n_eps):
        rewards = []
        relation1 = torch.Tensor((0, 1))
        relation2 = torch.Tensor((1, 0))


        for i_episode in range(n_eps):
            robot_state = self.env.getState()
            s0 = torch.Tensor(robot_state[0][0] + robot_state[0][1]) / 10.0
            s1 = torch.Tensor(robot_state[1][0] + robot_state[1][1]) / 10.0
            s2 = torch.Tensor(robot_state[2][0] + robot_state[2][1]) / 10.0
            s3 = torch.Tensor(robot_state[3][0] + robot_state[3][1]) / 10.0

            e10 = torch.cat((s1, s0, relation1)).unsqueeze(0).unsqueeze(0)
            e20 = torch.cat((s2, s0, relation2)).unsqueeze(0).unsqueeze(0)
            e30 = torch.cat((s3, s0, relation2)).unsqueeze(0).unsqueeze(0)

            e01 = torch.cat((s0, s1, relation1)).unsqueeze(0).unsqueeze(0)
            e21 = torch.cat((s2, s1, relation2)).unsqueeze(0).unsqueeze(0)
            e31 = torch.cat((s3, s1, relation2)).unsqueeze(0).unsqueeze(0)

            e02 = torch.cat((s0, s2, relation2)).unsqueeze(0).unsqueeze(0)
            e12 = torch.cat((s1, s2, relation2)).unsqueeze(0).unsqueeze(0)
            e32 = torch.cat((s3, s2, relation1)).unsqueeze(0).unsqueeze(0)

            e03 = torch.cat((s0, s3, relation2)).unsqueeze(0).unsqueeze(0)
            e13 = torch.cat((s1, s3, relation2)).unsqueeze(0).unsqueeze(0)
            e23 = torch.cat((s2, s3, relation1)).unsqueeze(0).unsqueeze(0)

            s0 = s0.unsqueeze(0)
            s1 = s1.unsqueeze(0)
            s2 = s2.unsqueeze(0)
            s3 = s3.unsqueeze(0)

            goal = torch.Tensor(self.env.generateGoal()).unsqueeze(1)/10.0
            goals = goal.numpy().tolist()
            print "goal: " + str(goal.numpy())


            self.ounoise.scale = (noise_scale - final_noise_scale) * max(0, exploration_end - i_episode) / exploration_end + final_noise_scale
            self.ounoise.reset()
            episode_reward = 0
            n_batch = updates_per_eps

            for i_step in range(n_steps):
                e0 = torch.cat((e10, e20, e30), dim=1)
                e1 = torch.cat((e01, e21, e31), dim=1)
                e2 = torch.cat((e02, e12, e32), dim=1)
                e3 = torch.cat((e03, e13, e23), dim=1)

                a0 = self.agent.select_action(e0, s0, goal[0], self.ounoise)
                a1 = self.agent.select_action(e1, s1, goal[1], self.ounoise)
                a2 = self.agent.select_action(e2, s2, goal[2], self.ounoise)
                a3 = self.agent.select_action(e3, s3, goal[3], self.ounoise)

                vel = tuple([tuple(a) for a in torch.cat((a0, a1, a2, a3), 0).numpy()])
                robot_state_, reward, done = self.env.step(vel, goals)
                episode_reward += reward

                esga0 = (e0, s0, goal[0], a0)
                esga1 = (e1, s1, goal[1], a1)
                esga2 = (e2, s2, goal[2], a2)
                esga3 = (e3, s3, goal[3], a3)


                robot_state = robot_state_
                s0 = torch.Tensor(robot_state[0][0] + robot_state[0][1]) / 10.0
                s1 = torch.Tensor(robot_state[1][0] + robot_state[1][1]) / 10.0
                s2 = torch.Tensor(robot_state[2][0] + robot_state[2][1]) / 10.0
                s3 = torch.Tensor(robot_state[3][0] + robot_state[3][1]) / 10.0

                e10 = torch.cat((s1, s0, relation1)).unsqueeze(0).unsqueeze(0)
                e20 = torch.cat((s2, s0, relation2)).unsqueeze(0).unsqueeze(0)
                e30 = torch.cat((s3, s0, relation2)).unsqueeze(0).unsqueeze(0)

                e01 = torch.cat((s0, s1, relation1)).unsqueeze(0).unsqueeze(0)
                e21 = torch.cat((s2, s1, relation2)).unsqueeze(0).unsqueeze(0)
                e31 = torch.cat((s3, s1, relation2)).unsqueeze(0).unsqueeze(0)

                e02 = torch.cat((s0, s2, relation2)).unsqueeze(0).unsqueeze(0)
                e12 = torch.cat((s1, s2, relation2)).unsqueeze(0).unsqueeze(0)
                e32 = torch.cat((s3, s2, relation1)).unsqueeze(0).unsqueeze(0)

                e03 = torch.cat((s0, s3, relation2)).unsqueeze(0).unsqueeze(0)
                e13 = torch.cat((s1, s3, relation2)).unsqueeze(0).unsqueeze(0)
                e23 = torch.cat((s2, s3, relation1)).unsqueeze(0).unsqueeze(0)

                s0 = s0.unsqueeze(0)
                s1 = s1.unsqueeze(0)
                s2 = s2.unsqueeze(0)
                s3 = s3.unsqueeze(0)

                es0_ = (e0, s0)
                es1_ = (e1, s1)
                es2_ = (e2, s2)
                es3_ = (e3, s3)

                reward = reward.tolist()
                r0 = reward[0]
                r1 = reward[1]
                r2 = reward[2]
                r3 = reward[3]

                self.pub.publish(r0)

                self.memory.push((esga0, es0_, r0))
                self.memory.push((esga1, es1_, r1))
                self.memory.push((esga2, es2_, r2))
                self.memory.push((esga3, es3_, r3))

                if any(done):
                    break

            self.memory.save()

            if len(self.memory) > batch_size * n_batch:
                n_batch += 6
                print "learning..."
                for _ in range(updates_per_eps):
                    batch = self.memory.sample(batch_size)
                    self.agent.update_parameters(batch)

            rewards.append(episode_reward)
            print("Episode: {}, noise: {}, return: {}, "
                  "average return: {}".format(i_episode, self.ounoise.scale,
                                              rewards[-1], np.mean(rewards[0][-100:])))

            print "random walking"
            self.env.reset()
            print '\n'

if __name__ == '__main__':
    rospy.init_node('game', anonymous=False)
    game = Game(env_id=1, n_robot=4)
    #game = Game()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS node game")