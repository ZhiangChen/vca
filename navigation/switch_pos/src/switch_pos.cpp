/// swtich position
/// Zhiang Chen, 3/2018
/// MIT License

#include <ros/ros.h> 
#include <iostream>
#include <switch_pos/robot_walk.h>

using namespace std;

int main(int argc, char **argv)
{
	ros::init(argc, argv, "switch_pos");
	ros::NodeHandle nh;
	Robot_walk Rw(&nh, "robot0");

	while(ros::ok())
	{
		Rw.random_walk(5);
		ROS_INFO("Done Random Walk");
	}
	return 0;
}



