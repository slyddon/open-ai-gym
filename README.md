# open-ai-gym

## Actions and observations for lunar lander

### Actions
Action is two floats [main engine, left-right engines].
* Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
* Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off

Default action is not continuous. The 4 discrete actions are:
* 0 - do nothing
* 1 - fire right
* 2 - fire main
* 3 - fire left

### Observations
Observation composed of
* pos.x
* pos.y
* vel.x
* vel.y
* lander.angle
* 20*lander.angularVelocity/FPS
* 1 if left leg touching ground else 0
* 1 if right leg touching ground else 0

### Reward
Reward = 0 - main_engine_use - side_engine_use
           - dist_from_landing_area - speed - angle
           + if_legs_touching - if_lost + if_won