"""
stateReader.py
Zhiang Chen
4/7/2018
"""

import rospy
from nav_msgs.msg import Odometry
from pyquaternion import Quaternion
import numpy as np

class robot:
	def __init__(self, robot_name):
		self.name = robot_name
		topic = robot_name + '/base_pose_ground_truth'
		self.sub = rospy.Subscriber(topic, Odometry, self.callback, queue_size=1)
		self.update = False
		self.got_data = False


	def callback(self, data):
		if not self.got_data:
			self.header_, self.position_, self.angle_ = self.extract(data)
			self.got_data = True
		else:
			self.header, self.position, self.angle = self.header_, self.position_, self.angle_
			self.header_, self.position_, self.angle_ = self.extract(data)
			self.velocity = self.get_vel()
			self.update = True

	def extract(self, data):
		header = data.header
		position = data.pose.pose.position
		quat = data.pose.pose.orientation
		angle = Quaternion(quat.w, quat.x, quat.y, quat.z).yaw_pitch_roll[0]
		return header, position, angle

	def get_vel(self):
		time = (self.header_.stamp.secs - self.header.stamp.secs) \
			   + (self.header_.stamp.nsecs - self.header.stamp.nsecs)/1000000000.0
		vx = (self.position_.x - self.position.x)/time
		vy = (self.position_.y - self.position.y)/time
		va = (self.angle_ - self.angle)/time
		return [vx, vy, va]

class stateReader:

	def __init__(self, num):
		names = ['robot_'+str(i) for i in range(num)]
		self.robots = [robot(name) for name in names]
		#self.test()

	def test(self):
		while not (rospy.is_shutdown()):
			for rob in self.robots:
				if rob.update:
					print rob.name
					print rob.velocity
					rob.update = False


if __name__ == '__main__':
	rospy.init_node('state_reader',anonymous=True)
	reader = stateReader(4)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down ROS node state_reader")
