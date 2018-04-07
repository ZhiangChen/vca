/// swtich position
/// Zhiang Chen, 3/2018
/// MIT License

#include <ros/ros.h> 
#include <iostream>
#include <switch_pos/robot_walk.h>
#include <math.h>

using namespace std;

#define R 0.5 //robot dimension

vector<geometry_msgs::Pose> random_walk(double sec, double dist, Robot_walk* rb0, Robot_walk* rb1, bool keep_walking = true);
void switch_pos(geometry_msgs::Pose goal0, geometry_msgs::Pose goal1, Robot_walk* rb0, Robot_walk* rb1);
double compute_dist(const geometry_msgs::Pose ps0, const geometry_msgs::Pose ps1);

int main(int argc, char **argv)
{
	ros::init(argc, argv, "switch_pos");
	ros::NodeHandle nh;
	Robot_walk Rw0(&nh, "robot0");
	Robot_walk Rw1(&nh, "robot1");
	vector<geometry_msgs::Pose> goals;
	goals.resize(2);


	while(ros::ok())
	{
		goals = random_walk(5, 2.5, &Rw0, &Rw1);
		ROS_INFO("random walk done.");
		switch_pos(goals[1], goals[0], &Rw0, &Rw1);
	}
	return 0;
}

void switch_pos(geometry_msgs::Pose goal0, geometry_msgs::Pose goal1, Robot_walk* rb0, Robot_walk* rb1)
{
	while(ros::ok())
	{
		if(!rb0->goal_walk(goal0))
			if ( !rb1->goal_walk(goal1))
				ROS_INFO("Positions Switching ...");

		ros::Duration(5.0).sleep();		
	}

}

vector<geometry_msgs::Pose> random_walk(double sec, double dist, Robot_walk* rb0, Robot_walk* rb1, bool keep_walking)
{
	vector<geometry_msgs::Pose> goals;
	goals.resize(2);
	
	double curDist = 0;
	while(curDist < dist && ros::ok())
	{
		ROS_INFO("Randomly Walking ...");
		rb0->random_walk(sec);
		goals[0] = rb0->getPose();
		goals[1] = rb1->getPose();
		curDist = compute_dist(goals[0], goals[1]);
		cout<<curDist;
		if (curDist > dist) break;

		rb1->random_walk(sec);
		goals[0] = rb0->getPose();
		goals[1] = rb1->getPose();
		curDist = compute_dist(goals[0], goals[1]);
		cout<<curDist;
	}
	
	if (keep_walking)
	{
		ROS_INFO("Keep Moving ...");
		vector<geometry_msgs::Pose> ps;
		ps.resize(2);

		curDist = 0;
		while (curDist < R && ros::ok())
		{
			rb0->random_walk(sec);
			ps[0] = rb0->getPose();
			curDist = compute_dist(ps[0], goals[0]);
		}

		curDist = 0;
		while (curDist < R && ros::ok())
		{
			rb1->random_walk(sec);
			ps[1] = rb0->getPose();
			curDist = compute_dist(ps[1], goals[1]);
		}
	}

	return goals;
	
}

double compute_dist(const geometry_msgs::Pose ps0, const geometry_msgs::Pose ps1)
{
	double x,y,z;
	x = ps0.position.x - ps1.position.x;
	y = ps0.position.y - ps1.position.y;
	z = ps0.position.z - ps1.position.z;
	return sqrt(x*x + y*y + z*z);
}

