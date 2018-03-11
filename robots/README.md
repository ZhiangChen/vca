# Turtlebot 3

1. [Turtlebot 3 e-Manual](http://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)

2. Gazebo Simulation
- launch one waffle  
```bash
roslaunch robots turtlebot3_world.launch
```  
- launch two waffles  
```bash
roslaunch robots two_waffles.launch
```  
Make sure that all topics and frames specified in URDF Plugins are without the /, otherwise they will be shared, e.g.:  
```
<topicName>imu</topicName>
<serviceName>imu_service</serviceName>
```

3. SLAM (see e-Manual)  
```bash
roslaunch robots turtlebot3_world.launch
export TURTLEBOT3_MODEL=waffle
roslaunch turtlebot3_slam turtlebot3_slam.launch
rosrun rviz rviz -d `rospack find turtlebot3_slam`/rviz/turtlebot3_slam.rviz
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
rosrun map_server map_saver -f ~/map
```

4. Navigation  
```bash
roslaunch robots env.launch
```  
[ROS namespaces](http://wiki.ros.org/Names) are crucial for multi-agent systems. Also, refer to [good materials by R. Yehoshua](http://u.cs.biu.ac.il/~yehoshr1/89-689/)
