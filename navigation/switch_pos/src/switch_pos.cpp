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
	Robot_walk Rw0(&nh, "robot0");
	Robot_walk Rw1(&nh, "robot1");
	geometry_msgs::Pose ps0 = Rw0.getPose();
	geometry_msgs::Pose ps1 = Rw1.getPose();

	while(ros::ok())
	{
		if(!Rw0.goal_walk(ps1))
			if ( !Rw1.goal_walk(ps0))
				ROS_INFO("switching positions...");

		ros::Duration(5.0).sleep();		
	}
	return 0;
}



