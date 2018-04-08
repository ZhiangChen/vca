#!/usr/bin/env python2

"""
randomWalk.py
Zhiang Chen
4/7/2018
"""

from stateReader import stateReader
import rospy
from graph_learning.msg import randomWalkMSG
from geometry_msgs.msg import Twist as Velocity
import random
import numpy as np
from math import *

group = [(0, 1), (2,3)]
# group_0 has robot_0, robot_1, and robot_0 is the leader;
# group_1 has robot_2, robot3, and robot_2 is the leader
robot_num = 4


class randomWalk:
    def __init__(self):
        self.states = stateReader(robot_num)
        self.forward = Velocity()
        self.forward.linear.x = 0.2
        self.backward = Velocity()
        self.backward.linear.x = -0.2
        self.turn = Velocity()
        self.turn.angular.z = 0.1
        self.stop = Velocity()
        self.sub = rospy.Subscriber('/random_group', randomWalkMSG, self.callback, queue_size=1)

    def callback(self, msg):
        if msg.groupID >= len(group) | msg.groupID < 0:
            rospy.logerr("Group ID is illegal")
        else:
            t = msg.time
            leaderID = group[msg.groupID][0]
            leader = self.states.robots[leaderID]
            step = t/200.0
            sum_t = 0
            while sum_t < t:
                if leader.obstacle is None:
                    return
                elif not leader.obstacle:
                    leader.pub.publish(self.forward)
                    sum_t += step
                    rospy.sleep(step)
                else:
                    n = random.randint(1,50)
                    for _ in range(n):
                        leader.pub.publish(self.turn)
                        sum_t += step
                        rospy.sleep(step)
            leader.pub.publish(self.stop)

            followersID = group[msg.groupID][1:]
            followers = [self.states.robots[i] for i in followersID]
            for follower in followers:
                x = leader.position_.x - follower.position_.x
                y = leader.position_.y - follower.position_.y
                v = np.array([x,y])
                dist = np.linalg.norm(v)
                v =  v/dist
                if v[1] > 0:
                    angle = np.arccos(v[0])
                else:
                    angle = -np.arccos(v[0])
                while abs(angle - follower.angle) > 0.05:
                    follower.pub.publish(self.turn)
                follower.pub.publish(self.stop)

                while dist > 1.52:
                    follower.pub.publish(self.forward)
                    x = leader.position_.x - follower.position_.x
                    y = leader.position_.y - follower.position_.y
                    v = np.array([x, y])
                    dist = np.linalg.norm(v)
                    print dist
                    while dist < 1.48:
                        follower.pub.publish(self.backward)
                        follower.pub.publish(self.forward)
                        x = leader.position_.x - follower.position_.x
                        y = leader.position_.y - follower.position_.y
                        v = np.array([x, y])
                        dist = np.linalg.norm(v)
                follower.pub.publish(self.stop)





if __name__ == '__main__':
    rospy.init_node('random_walk', anonymous=True)
    walker = randomWalk()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS node random_walk")