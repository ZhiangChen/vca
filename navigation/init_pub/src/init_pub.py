#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped as PCS

def pub():
	pub0 = rospy.Publisher("robot0/initialpose", PCS, queue_size = 1)
	pub1 = rospy.Publisher("robot1/initialpose", PCS, queue_size = 1)
	rospy.init_node('init_pub',anonymous = True)
	rate = rospy.Rate(20) 
	pcs = PCS()
	pcs.header.frame_id = "map"
	for _ in range(10):
		pcs.pose.pose.position.x = -2.0
		pcs.pose.pose.position.y = -0.5
		pcs.pose.pose.position.z = 0
		pcs.pose.pose.orientation.x = 0.0
		pcs.pose.pose.orientation.y = 0.0
		pcs.pose.pose.orientation.z = 0.0
		pcs.pose.pose.orientation.w = 1.0
		pub0.publish(pcs)

		pcs.pose.pose.position.x = 2.0
		pcs.pose.pose.position.y = -0.5
		pcs.pose.pose.position.z = 0
		pcs.pose.pose.orientation.x = 0.0
		pcs.pose.pose.orientation.y = 0.0
		pcs.pose.pose.orientation.z = 1.0
		pcs.pose.pose.orientation.w = 0.0
		pub1.publish(pcs)
		rate.sleep()
	rospy.loginfo("InitialPose Done!")
	
	

if __name__ == "__main__":
	try:
		pub()
	except rospy.ROSInterruptException:
		pass

