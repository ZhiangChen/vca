#!/usr/bin/env python2

"""
randomWalk.py
Zhiang Chen
4/7/2018
"""

from stateReader import stateReader
import rospy
from graph_learning.srv import randomWalkSRV
from graph_learning.srv import randomWalkSRVResponse
from geometry_msgs.msg import Twist as Velocity
import random
import numpy as np
from math import *
import time

"""PARAMETERS TO TUNE"""
group = [(0, 1), (2,3)]
# group_0 has robot_0, robot_1, and robot_0 is the leader;
# group_1 has robot_2, robot3, and robot_2 is the leader

class randomWalk(object):
    def __init__(self, robot_num=4, dist = 1.5):
        self._dist_between_robots = dist
        self.states = stateReader(robot_num)
        self.forward = Velocity()
        self.forward.linear.x = 0.2
        self.backward = Velocity()
        self.backward.linear.x = -0.2
        self.turn = Velocity()
        self.turn.angular.z = 0.1
        self.stop = Velocity()
        self.service = rospy.Service('/random_group', randomWalkSRV, self.callback)
        rospy.loginfo("Random Walker Initialized")
        rospy.sleep(1.0)

    def callback(self, srv):
        if srv.groupID >= len(group) | (srv.groupID < 0):
            rospy.logerr("Group ID is illegal")
            randomWalkSRVResponse(False)
        else:
            t = srv.time
            leaderID = group[srv.groupID][0]
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

            followersID = group[srv.groupID][1:]
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

                while dist > (self._dist_between_robots + 0.02):
                    follower.pub.publish(self.forward)
                    x = leader.position_.x - follower.position_.x
                    y = leader.position_.y - follower.position_.y
                    v = np.array([x, y])
                    dist = np.linalg.norm(v)
                    #print dist
                    while dist < (self._dist_between_robots - 0.02):
                        follower.pub.publish(self.backward)
                        follower.pub.publish(self.forward)
                        x = leader.position_.x - follower.position_.x
                        y = leader.position_.y - follower.position_.y
                        v = np.array([x, y])
                        dist = np.linalg.norm(v)
                follower.pub.publish(self.stop)
            return randomWalkSRVResponse(True)


    def action(self, group_id, vel):
        """
        :param group_id: group_id
        :param vel: velocities of all robots in the group
        :return: None if error occurs
        """
        if group_id >= len(group) | (group_id < 0):
            rospy.logerr("Group ID is illegal")
            return None
        else:
            if len(vel) != len(group[group_id]):
                rospy.logerr("Velocities cannot match group")
                return None
            else:
                #t1 = time.time()
                leaderID = group[group_id][0]
                leader = self.states.robots[leaderID]
                followersID = group[group_id][1:]
                followers = [self.states.robots[i] for i in followersID]
                vel = [self._converter(v) for v in vel]
                #print time.time() - t1
                #t2 = time.time()
                for _ in range(10):
                    leader.pub.publish(vel[0])
                    #print leader.getVelocity()
                    """print consumes a lot of time"""
                    for i,follower in enumerate(followers):
                        follower.pub.publish(vel[i+1])
                        #print follower.getVelocity()
                    time.sleep(0.01) # time.sleep is better than rospy.sleep
                #print time.time() - t2

    def getState(self):
        """
        getState is mainly limited to the frequency of state publishers. The reader process is actually very fast.
        :return:
        """
        return tuple([robot.getState() for robot in self.states.robots])


    def _converter(self, v):
        vel = Velocity()
        vel.linear.x = v[0]
        vel.angular.z = v[1]
        return vel





if __name__ == '__main__':
    rospy.init_node('random_walk', anonymous=False)
    walker = randomWalk()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS node random_walk")
