### Dependencies
1. pyquaternion:  
http://kieranwynn.github.io/pyquaternion/

2. rl:  
https://github.com/ikostrikov/pytorch-ddpg-naf  
keep the repo in the directory "vca/graph_learning/src/" and change name as "rl". Also, add \__init__.py in the repo to make it a library.

### Random Walk
The leader randomly walks and then the followers chase.
``` bash
roscore
rosrun stage_ros stageros $(rospack find robots)/world/stage_sim/simple_world.world  
python env.py -r
```
