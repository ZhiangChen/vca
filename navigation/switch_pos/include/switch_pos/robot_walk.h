/// robot_walk.h
/// Zhiang Chen, 3/2018

/*The MIT License (MIT)
Copyright (c) 2016 Zhiang Chen
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#ifndef RANDOM_WALK_H_
#define RANDOM_WALK_H_

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

using namespace std;

#define ANGLE 45.0
#define DIST 0.5
#define VEL_RATE 20



class Robot_walk
{
public:
	Robot_walk(ros::NodeHandle* nh, string robot_name);
	void random_walk(double sec);
	bool goal_walk(geometry_msgs::Pose goal);
	geometry_msgs::Pose getPose();
	

private:
	ros::NodeHandle nh_;
	string robot_name_;
	string vel_topic_;
	string scan_topic_;
	string r_scan_topic_;
	string odom_topic_;
	string goal_topic_;

	ros::Subscriber scan_sub_;
	ros::Subscriber odom_sub_;
	ros::Publisher cmd_vel_pub_;
	ros::Publisher r_scan_pub_;
	ros::Publisher goal_pub_;
	MoveBaseClient ac_;

	bool isObstacle_;
	bool gotScan_;
	bool gotOdom_;
	geometry_msgs::Pose curPose_;

    geometry_msgs::Twist moveForwardCommand_;
    geometry_msgs::Twist turnLeftCommand_;
	geometry_msgs::Twist turnRightCommand_;
	geometry_msgs::Twist stopCommand_;
	

	void readScanCallback(const sensor_msgs::LaserScan::ConstPtr &scan);
	void readOdomCallback(const nav_msgs::Odometry odom);
	
};

#endif

