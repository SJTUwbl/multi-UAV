# UAV swarm confrontation Environment

A simple UAV swarm confrontation environment with a continuous observation and continuous action space, along with some basic simulated physics, which is based on the multi-agent particle environment.
![image](UAV%20swarm%20confrontation.png)

## Observation space
- Own feature: position, velocity, speed, yaw, roll
- Ally feature: relative position, distance, relative yaw, roll  
- Adversary feature: relative position, distance, angle of attack, relative yaw, roll
- Ally alive mask
- Adversary alive mask

## Action space
- Acceleration: [-1, 1]
- Roll rate: [-1, 1]

## Requirements:
- python==3.8.8
- gym==0.21.0
- numpy==1.20.1
- pyglet==1.5.0
- six==1.16.0
