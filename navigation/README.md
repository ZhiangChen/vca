# Navigation

1. init_pub  
init_pub initializes the robots' poses to assist amcl localizers.  
```bash
python init_pub.py
```

2. switch_pos  
switch_pub has a library [Robot_walk.h](https://github.com/ZhiangChen/vca/blob/master/navigation/switch_pos/include/switch_pos/robot_walk.h), which 
provides two main methods - random_walk and goal_walk. The node switch_pos switches the robots' poses after some random walk.  
```bash
rosrun switch_pos switch_pos
```
