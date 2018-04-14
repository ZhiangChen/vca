### Dependencies
1. pyquaternion:  
http://kieranwynn.github.io/pyquaternion/

### Random Walk
The leader randomly walks and then the followers chase.
``` bash
roscore
rosrun stage_ros stageros $(rospack find robots)/world/stage_sim/simple_world.world  
python env.py -r
```
