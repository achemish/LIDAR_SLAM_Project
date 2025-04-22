"""
LIDAR SLAM Project - Winter 2024
Author: Aram Chemishkian (achemish@gmail.com)
main.py: 
    This script executes part 1 through 4 of PR2. Given encoder, IMU, & LIDAR data from a ground
    robot, the script processes the data and performs SLAM via pose-graph optimization to recover
    the robot trajectory and build an occupancy grid of the surrounding environment.

Future Improvements:
    - Filter the LIDAR data (possibly IMU/Encoders too) before computation for better results
    - Improve efficiency of the scan matching algorithm for faster computation
    - Make ICP algorithm more robust to bad initializations
    - Tune Pose Graph for better final estimate
    - Re-write with OOP approach
    - Clean up library use/imports
    - Add full docstrings w/inputs & outputs, dimensions, to all functions
"""
from icp_warm_up.utils import *
from code.given_utils import *
from icp import *
from constants import *
from load_data import *
from custom_utils import *



if __name__ == '__main__':
    print("PR2: Loading data...")

    ### Configs
    run_part3 = False # Occupancy Grid based on LIDAR Odometry only
    run_part4 = True # GTSAM Pose Slam (Trajectory + Updated Occupancy Grid)

    dataset_index = 21  # select 20 or 21
    dataset = read_data(dataset_index)

    # Downsampling Configs 
    # (warning: downsampling severely impacts quality of scan matching and GTSLAM)
    sample_rate = 1 #use every nth sample (set to 1 to use every sample, >1 for faster computation but worse result)
    num_scans_config = dataset.lidar_ranges.shape[1]-1 # Set num of LIDAR scans to use. Leave as
                                                       # "lidar_ranges.shape[1]-1" to use all. Helpful
                                                       # to set to a lower number for faster debugging

    ### Part 1: Apply differential diff model to estimate odometry from IMU/Encoders
    x_model, y_model, pose_model = perform_diff_drive_model_odometry(dataset)

    ### Part 2/3/4 Prep: Extract pointclouds (in local coords) from range data
    print("Extracting pointclouds")
    pointcloud_dataset = extract_pointcloud_dataset(dataset.lidar_ranges,dataset.lidar_angle_increment)
    print("Pointcloud extraction done")


    ### Part 2: Scan Matching (LIDAR-only trajectory, no odometry)
    
    #run scan-matching to estimate odometry from LIDAR data
    x_lidar, y_lidar, pose_lidar = perform_LIDAR_odometry(dataset,dataset_index, pointcloud_dataset,pose_model, num_scans_config,sample_rate)

    #plot odometry estimates
    trajectory_estimates = ((x_model,y_model),
    (x_lidar[:num_scans_config//sample_rate],y_lidar[:num_scans_config//sample_rate]))
    trajectory_estimate_labels = ['Motion Model Estimate', 'Scan-Match Estimate']
    plot_title = 'Motion Model vs Scan-Match Trajectory:'
    plot_trajectories(trajectory_estimates, trajectory_estimate_labels, plot_title)


    print("Done plotting lidar trajectory")



    ### Part 3 - Occupancy Grid based on LIDAR Odometry only
    if(run_part3):
        width = 50 #m
        height = 50 #m
        grid_res = 0.1 #m/cell
        lidar_trust = 80 #%trust
        occupancy_grid = build_occupancy_grid(pose_lidar, pointcloud_dataset, num_scans_config, sample_rate, lidar_trust, width, height, grid_res)

        #plot resulting grid
        trajectory_estimates = ((x_model,y_model),
            (x_lidar[:num_scans_config//sample_rate],y_lidar[:num_scans_config//sample_rate]))
        trajectory_estimate_labels = ['Motion Model Estimate', 'Scan-Match Estimate']
        plot_SLAM(trajectory_estimates, trajectory_estimate_labels, occupancy_grid, width, grid_res)
        

    ### Part 4 - GTSAM Pose Slam (Trajectory + Updated Occupancy Grid)
    if(run_part4):
        x_gtsam, y_gtsam, pose_gtsam = perform_pose_graph_odometry(dataset_index, pose_lidar, pointcloud_dataset, num_scans_config, sample_rate)

        #recompute occupancy grid with pose SLAM output
        width = 50 #m
        height = 50 #m
        grid_res = 0.1 #m/cell
        lidar_trust = 80 #%trust
        occupancy_grid = build_occupancy_grid(pose_gtsam, pointcloud_dataset, num_scans_config, sample_rate, lidar_trust, width, height, grid_res)

        # plot resulting grid
        trajectory_estimates = ((x_model,y_model),
            (x_lidar[:num_scans_config//sample_rate],y_lidar[:num_scans_config//sample_rate]),
            (x_gtsam[:num_scans_config//sample_rate],y_gtsam[:num_scans_config//sample_rate]))
        trajectory_estimate_labels = ['Motion Model Estimate', 'Scan-Match Estimate', 'PoseSLAM Estimate']
        plot_title = 'Motion Model vs Scan-Match vs PoseSLAM Trajectory:'
        plot_trajectories(trajectory_estimates, trajectory_estimate_labels, plot_title)
        plot_SLAM(trajectory_estimates, trajectory_estimate_labels,occupancy_grid,  width, grid_res)
