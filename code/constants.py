"""
LIDAR SLAM Project - Winter 2024
Author: Aram Chemishkian (achemish@gmail.com)
constants.py: 
    This file declares constants based on the robot/sensor attributes for use
    in various computations throughout this project.
"""
import math

### Constants ###
NUM_POINTS_PER_LIDAR_SCAN = 1081
MIN_LIDAR_ANGLE_DEGREES = -135.0 #270 deg FOV LIDAR, range -135 deg to 135 deg
MIN_LIDAR_ANGLE_RADS = MIN_LIDAR_ANGLE_DEGREES*math.pi/180.0 
METERS_PER_ENCODER_TICK = 0.0022  # meters/tick (360 ticks, 0.254m wheel diam)
ENCODER_RATE = 40  # hz