#!/usr/bin/env python2

"""
stateReader.py
Zhiang Chen
4/7/2018
"""

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from pyquaternion import Quaternion
from geometry_msgs.msg import Twist as Velocity
import numpy as np


class robot:
    def __init__(self, robot_name):
        """
        self.name: robot name, str
        self.position: robot position, ros position
        self.angle: absolute robot angle, float
        self.velocity: absolute robot velocity, list, [v_x, v_y, v_angle]
        self.update: boolean parameter controlling data updating
        """
        self.update = False
        self.got_data = False
        self.obstacle = None

        self.name = robot_name
        topic = robot_name + '/base_pose_ground_truth'
        topic1 = robot_name + '/base_scan'
        topic_vel = robot_name + '/cmd_vel'
        self.sub = rospy.Subscriber(topic, Odometry, self.callback, queue_size=1)
        self.sub1 = rospy.Subscriber(topic1, LaserScan, self.callback1, queue_size=1)
        self.pub = rospy.Publisher(topic_vel, Velocity, queue_size=1)
        # self.pub = rospy.Publisher('lidar', LaserScan, queue_size = 1)
        rospy.loginfo("State Reader %s Initialized" % robot_name)

    def callback(self, data):
        if not self.got_data:
            self.header_, self.position_, self.angle_ = self._extract(data)
            self.got_data = True
        else:
            self.header, self.position, self.angle = self.header_, self.position_, self.angle_
            self.header_, self.position_, self.angle_ = self._extract(data)
            self.velocity = self._get_vel()
            self.update = True

    def callback1(self, data):
        nm = 110
        mid = len(data.ranges) / 2
        index = [mid + i for i in range(-nm, nm)]
        scans = [data.ranges[i] for i in index]
        # data.ranges = scans
        # self.pub.publish(data)
        if min(scans) < 1.0:
            self.obstacle = True
        else:
            self.obstacle = False

    def getState(self):
        while not self.update:
            None
        self.update = False
        return self.position_, self.angle_, self.velocity

    def getVelocity(self):
        while not self.update:
            None
        self.update = False
        return self.velocity


    def _extract(self, data):
        header = data.header
        position = data.pose.pose.position
        quat = data.pose.pose.orientation
        angle = Quaternion(quat.w, quat.x, quat.y, quat.z).yaw_pitch_roll[0]
        return header, position, angle

    def _get_vel(self):
        time = (self.header_.stamp.secs - self.header.stamp.secs) \
               + (self.header_.stamp.nsecs - self.header.stamp.nsecs) / 1000000000.0
        vx = (self.position_.x - self.position.x) / time
        vy = (self.position_.y - self.position.y) / time
        va = (self.angle_ - self.angle) / time
        return [vx, vy, va]


class stateReader:
    def __init__(self, num):
        names = ['robot_' + str(i) for i in range(num)]
        self.robots = [robot(name) for name in names]

    # self.test()

    def test(self):
        while not (rospy.is_shutdown()):
            for rob in self.robots:
                if rob.update:
                    print rob.name
                    print rob.velocity
                    rob.update = False


if __name__ == '__main__':
    rospy.init_node('state_reader', anonymous=False)
    reader = stateReader(4)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS node state_reader")
