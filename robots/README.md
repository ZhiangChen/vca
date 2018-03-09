# Turtlebot 3

1. [Turtlebot 3 e-Manual](http://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)

2. Gazebo Simulation
- launch one waffle  
`bash
roslaunch robots turtlebot3_world.launch
`
- launch two waffles  
`bash
roslaunch robots two_waffles.launch
`  
(igonre the imu error since it is not used)

3. SLAM (see e-Manual)  
`bash
roslaunch robots turtlebot3_world.launch
export TURTLEBOT3_MODEL=waffle
roslaunch turtlebot3_slam turtlebot3_slam.launch
rosrun rviz rviz -d `rospack find turtlebot3_slam`/rviz/turtlebot3_slam.rviz
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
rosrun map_server map_saver -f ~/map
`


