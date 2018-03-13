/// robot_walk.cpp
/// Zhiang Chen, 3/2018
/// MIT License

#include <switch_pos/robot_walk.h>
#include <iostream>

using namespace std;

Robot_walk::Robot_walk(ros::NodeHandle* nh, string robot_name): nh_(*nh), robot_name_(robot_name)
{
	vel_topic_ = robot_name_ + "/cmd_vel";
	scan_topic_ = robot_name_ + "/scan";
	r_scan_topic_ = robot_name_ +"/r_scan";

	scan_sub_ = nh_.subscribe(scan_topic_, 1, &Robot_walk::readScanCallback, this);
	r_scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>(r_scan_topic_, 1, true);
	cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>(vel_topic_, 1, true);

	moveForwardCommand_.linear.x = 0.2;
	turnLeftCommand_.angular.z = 0.4;
	turnRightCommand_.angular.z = -0.4;
	srand(time(NULL));
	gotData_ = false;

	ROS_INFO("Robot_walk initialized!");
}

void Robot_walk::readScanCallback(const sensor_msgs::LaserScan::ConstPtr &scan)
{
	sensor_msgs::LaserScan r_scan;
	r_scan.header = scan->header;
	r_scan.angle_increment = scan->angle_increment;
	r_scan.angle_min = -ANGLE/180.0*3.14;
	r_scan.angle_max = ANGLE/180.0*3.14;
	r_scan.range_min = scan->range_min;
	r_scan.range_max = scan->range_max;
	int n = r_scan.angle_max / r_scan.angle_increment;
	int mn = scan->angle_max / scan->angle_increment;
	r_scan.ranges.resize(2*n);
	r_scan.intensities.resize(2*n);
	for (int i=0; i<2*n; i++)
	{
		if (i<n)
		{
			r_scan.ranges[i] = scan->ranges[mn-n+i];
			r_scan.intensities[i] = scan->intensities[mn-n+i];
		}
		else
		{
			r_scan.ranges[i] = scan->ranges[i-n];
			r_scan.intensities[i] = scan->intensities[i-n];
		}
	}

	isObstacle_ = false;
	for (int i=0; i<2*n; i++)
	{
		if(r_scan.ranges[i] < DIST)
		{
			isObstacle_ = true;
			break;
		}
	}

	r_scan_pub_.publish(r_scan);
	gotData_ = true;
}

void Robot_walk::random_walk(double sec)
{
	ros::Rate r(VEL_RATE);
	double incrt = 1.0/VEL_RATE;
	double t = 0;
	while(t < sec)
	{
		
		while(!gotData_)
		{
			ros::spinOnce();
			ros::Duration(0.05).sleep();
			t +=0.05;
		}
		gotData_ = false;
		if (isObstacle_) 
		{
			if(rand()/RAND_MAX<0.5)
			{
				ROS_INFO("Turning left");
				cmd_vel_pub_.publish(turnLeftCommand_);
			}
			else
			{
				ROS_INFO("Turning right");
				cmd_vel_pub_.publish(turnRightCommand_);
			}
        } 
		else 
		{
			ROS_INFO("Moving forward");
			cmd_vel_pub_.publish(moveForwardCommand_);
        }
		r.sleep();
		t += incrt;
	}
}



