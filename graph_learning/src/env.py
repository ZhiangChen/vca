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

env_id = 1
# 0: two mobile robots
# 1: four mobile robots

class Env(randomWalk):
    def __init__(self, env_id):
        super(Env, self).__init__()
        self.env_id = env_id
        #self.pub = rospy.Publisher('/random_group', randomWalkSRV, queue_size=1)
        rospy.wait_for_service('/random_group', timeout=5.0)
        rospy.loginfo("Env Initialized")

        if len(sys.argv) == 2:
            if sys.argv[1] == '-r':
                self.reset()
                rospy.loginfo("Done")

            if sys.argv[1] == '-t':
                self.action(0, ((0.2,0.0),(0.2,0.2)))


    def reset(self):
        if self.env_id == 0:
            group_id = 0
            time = 5
            try:
                self.clinet = rospy.ServiceProxy('/random_group', randomWalkSRV)
                return self.clinet(group_id, time)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
                return None

        elif self.env_id == 1:
            group_id = 0
            time = 5
            try:
                self.clinet = rospy.ServiceProxy('/random_group', randomWalkSRV)
                self.clinet(group_id, time)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
                return None

            group_id = 1
            time = 5
            try:
                self.clinet = rospy.ServiceProxy('/random_group', randomWalkSRV)
                return self.clinet(group_id, time)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
                return None


    def step(self, action):
        if self.env_id == 0:
            pass
        elif self.env_id == 1:
            pass

    def calcReward(self):
        if self.env_id == 0:
            pass
        elif self.env_id == 1:
            pass





if __name__ == '__main__':
    rospy.init_node('env', anonymous=False)
    env = Env(env_id)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS node env")
