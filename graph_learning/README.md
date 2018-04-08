### Dependencies
1. pyquaternion:  
http://kieranwynn.github.io/pyquaternion/

### Random Walk
The leader randomly walks and then the followers chase.
``` bash
roscore
rosrun stage_ros stageros $(rospack find robots)/world/stage_sim/simple_world.world  
python randomWalk.py
rostopic pub -r 1 random_group graph_learning/randomWalkMSG '{groupID: 0, time: 5}'
```
