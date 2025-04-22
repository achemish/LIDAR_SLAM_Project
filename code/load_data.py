"""
LIDAR SLAM Project - Winter 2024
Author: Aram Chemishkian (achemish@gmail.com)
load_data.py: 
    This file implements helper functions to handle loading and processing 
    the dataset (containing IMU, LIDAR, and Encoder data from the robot)
"""

from dataclasses import dataclass

import numpy as np

from constants import *

### Dataset Class
@dataclass
class SensorDataset:
    encoder_counts: np.ndarray
    encoder_stamps: np.ndarray
    lidar_angle_min: np.ndarray
    lidar_angle_max: np.ndarray
    lidar_angle_increment: np.ndarray
    lidar_range_min: np.ndarray
    lidar_range_max: np.ndarray
    lidar_ranges: np.ndarray
    lidar_stamps: np.ndarray
    imu_angular_velocity: np.ndarray
    imu_linear_acceleration: np.ndarray
    imu_stamps: np.ndarray
    disp_stamps: np.ndarray
    rgb_stamps: np.ndarray


### Load Helpers
def read_data(dataset_index):
    """Given the dataset index, reads in the corresponding IMU/Encoder/LIDAR data files
    """
    assert (dataset_index == 20 or dataset_index == 21)
   
    with np.load("../data/SensorData/Encoders%d.npz" % dataset_index) as data:
        encoder_counts = data["counts"]  # 4 x n encoder counts
        encoder_stamps = data["time_stamps"]  # encoder time stamps

    with np.load("../data/SensorData/Hokuyo%d.npz" % dataset_index) as data:
        lidar_angle_min = data["angle_min"]  # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"]  # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"]  # angular distance between measurements [rad]
        lidar_range_min = data["range_min"]  # minimum range value [m]
        lidar_range_max = data["range_max"]  # maximum range value [m]
        lidar_ranges = data["ranges"]  # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

    with np.load("../data/SensorData/Imu%d.npz" % dataset_index) as data:
        imu_angular_velocity = data["angular_velocity"]  # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"]  # accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

    with np.load("../data/SensorData/Kinect%d.npz" % dataset_index) as data:
        disp_stamps = data["disparity_time_stamps"]  # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"]  # acquisition times of the rgb images
    return SensorDataset(
        encoder_counts=encoder_counts,
        encoder_stamps=encoder_stamps,
        lidar_angle_min=lidar_angle_min,
        lidar_angle_max=lidar_angle_max,
        lidar_angle_increment=lidar_angle_increment,
        lidar_range_min=lidar_range_min,
        lidar_range_max=lidar_range_max,
        lidar_ranges=lidar_ranges,
        lidar_stamps=lidar_stamps,
        imu_angular_velocity=imu_angular_velocity,
        imu_linear_acceleration=imu_linear_acceleration,
        imu_stamps=imu_stamps,
        disp_stamps=disp_stamps,
        rgb_stamps=rgb_stamps
    )
   
### Processing Helpers
def extract_pointcloud_single_scan(lidar_scan, lidar_angle_increment):
    """Take a single LIDAR scan and extract the points in local coord frame"""
    assert(lidar_scan.shape[0] == NUM_POINTS_PER_LIDAR_SCAN)
    pointcloud = np.zeros((3,NUM_POINTS_PER_LIDAR_SCAN)) #z coords are always 0 in local coord frame (because 2D LIDAR)
    #iterate through each range point, from -135 to 135 degrees CCW
    for i,range in enumerate(lidar_scan):
        theta = MIN_LIDAR_ANGLE_RADS+lidar_angle_increment*i
        pointcloud[0,i] = range*math.cos(theta) #x coord
        pointcloud[1,i] = range*math.sin(theta) #y coord

    return pointcloud

def extract_pointcloud_dataset(lidar_ranges,lidar_angle_increment):
    """Take a dataset of LIDAR scans and extract a list of corresponding pointclouds"""
    pointcloud_dataset = np.zeros((3,lidar_ranges.shape[0],lidar_ranges.shape[1]))
    for scan_index in range(lidar_ranges.shape[1]):
        pointcloud_dataset[:,:,scan_index] = extract_pointcloud_single_scan(
            lidar_ranges[:,scan_index], lidar_angle_increment[0,0])
    return pointcloud_dataset
