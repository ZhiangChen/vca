#!/usr/bin/env python2

"""
env.py
Zhiang Chen
4/13/2018
"""

from graph_learning.srv import randomWalkSRV
from randomWalk import *
import rospy
import sys
import numpy as np

env_id = 0
# 0: two mobile robots
# 1: four mobile robots
robot_num = 2

DIST = 1.5
# this distance should be consistant with the distance in randomWalk.callback

class Env(randomWalk):
    def __init__(self, env_id, robot_num=4):
        super(Env, self).__init__(robot_num)
        self.env_id = env_id
        #self.pub = rospy.Publisher('/random_group', randomWalkSRV, queue_size=1)
        rospy.wait_for_service('/random_group', timeout=5.0)
        rospy.loginfo("Env Initialized")

        if len(sys.argv) == 2:
            if sys.argv[1] == '-r':
                self.reset()
                rospy.loginfo("Done")

            if sys.argv[1] == '-t':
                for _ in range(5):
                    self.action(group_id=0,vel=((0.2,0.2),(0.1,0)))
                    print self.getState()
                    print '\n'

            if sys.argv[1] == '-t1':
                self.step(vel=((0.2,0.2),(0.1,0)), goal=((0.5,0.5),(0.5,0.5)))
                state1 = self.getState()
                for _ in range(5):
                    self.action(group_id=0, vel=((0.2, 0.2), (0.1, 0)))
                state2 = self.getState()
                print self._rewardFnc2(self._center(state1),self._center(state2))

    def reset(self):
        if self.env_id == 0:

            try:
                self.clinet = rospy.ServiceProxy('/random_group', randomWalkSRV)
                if self.clinet(group_id=0, time=5):
                    return self.states.robots
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
                return None

        elif self.env_id == 1:
            try:
                self.clinet = rospy.ServiceProxy('/random_group', randomWalkSRV)
                self.clinet(group_id=0, time=5)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
                return None

            try:
                self.clinet = rospy.ServiceProxy('/random_group', randomWalkSRV)
                return self.clinet(group_id=1, time=5)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
                return None


    def step(self, vel, goal):
        """
        :param vel: velocities of all robots, tuple
        :param goal: goals of all robots, tuple
        :return: updated states, tuple; rewards for all robots, tuple; termination status, bool.
        """
        if self.env_id == 0:
            self.action(group_id=0, vel=vel)
            state = self.getState()
            rewards = self.calcReward(state, goal)


        elif self.env_id == 1:
            pass

    def calcReward(self, state, goal):
        """
        :param state: states of robots in ONE group, tuple
        :param goal: goals of robots in ONE group, tuple
        :return: rewards for all the robots, tuple; termination status, bool;
        """
        if self.env_id == 0:
            pass
        elif self.env_id == 1:
            pass



    def _rewardFnc1(self, state):
        """
        rewardFnc1 computes the reward between two robots expected constant distance. See rewardFnc1 in folder img
        :param state: state of a group, tuple
        :return: reward, float
        """
        d = -abs(self._dist_in_group(state) - DIST)
        return min(0, 100**d-1)


    def _rewardFnc2(self, c1, c2):
        """
        rewardFnc2 computes the reward depending on current state center and goal center. This aims at the final state.
        :param c1: current center, numpy.array
        :param c2: goal center, numpy.array
        :return: reward, float
        """
        if np.linalg.norm(c1-c2) < 0.1:
            return 1.0
        else:
            return 0.0


    def _rewardFnc3(self, c1, c2):
        """
        this aims at the progress state
        :param c1: current center, numpy.array
        :param c2: goal center, numpy.array
        :return: reward, float
        """





    def _center(self, state):
        points = np.asarray(state)[:, 0, :2]
        return np.average(points, axis=0)

    def _dist_in_group(self, gstate):
        """
        compute the distance between two robots in a group. The group should only have two robots.
        :param state: states of robots, tuple.
        :return: distance, float; None if failed
        """
        if len(gstate) == 2:
            points = np.asarray(gstate)[:, 0, :2]
            return np.linalg.norm(points[0]-points[1])
        else:
            rospy.logerr("failed distance for group with incorrect number of robots")
            return None

    def _dist_between_group(self, state1, state2):
        """
        center distance between two groups
        :param state1: state of group1, tuple
        :param state2: state of group2, tuple
        :return: distance, float
        """
        c1 = self._center(state1)
        c2 = self._center(state2)
        return np.linalg.norm(c1 - c2)






if __name__ == '__main__':
    rospy.init_node('env', anonymous=False)
    env = Env(env_id, robot_num)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS node env")
