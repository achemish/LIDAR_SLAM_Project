"""
LIDAR SLAM Project - Winter 2024
Author: Aram Chemishkian (achemish@gmail.com)
custom_utils.py: 
    This file implements helper functions for performing odometry estimation, LIDAR-based trajectory
    estimation, occupancy grid mapping, and the combined LIDAR/Odometry SLAM. Possibly should be
    merged with the pr2_utils.py file.
"""
import gtsam 
import matplotlib as mpl
from matplotlib import pyplot as plt

from code.given_utils import *
from constants import *
from icp import *


### Odometry Helpers (Part 1)

def perform_diff_drive_model_odometry(dataset):
    """
    Use IMU & wheel encoder sensor data with differential drive motion model to estimate odometry 
    """
    # Compute linear velocity
    wheel_vel = dataset.encoder_counts * METERS_PER_ENCODER_TICK * ENCODER_RATE #compute velocity for each wheel
    avg_wheel_vel = np.sum(wheel_vel, axis=0) / 4   # avg wheel velocities together

    #apply diff model
    yaw_vel = dataset.imu_angular_velocity[2, :] # discard roll and pitch (robot operates in 2D, yaw only)
    state = np.zeros((3, dataset.imu_stamps.shape[0]))  # robot state (x,y,yaw)
    pose = np.zeros((4,4,dataset.imu_stamps.shape[0]))  # 4x4 3D pose (3x3 Rot matrix, 3x1 position)
    pose[:,:,0] = np.eye(4) # set origin for first timestamp

    #iterate through each IMU read and apply diff model
    for t in range(dataset.imu_stamps.shape[0] - 1):
        T_t = dataset.imu_stamps[t + 1] - dataset.imu_stamps[t]

        # find nearest encoder timestamp
        diff = dataset.encoder_stamps - dataset.imu_stamps[t]
        ts_index = np.argmin(abs(diff))
        v_t = avg_wheel_vel[ts_index]

        # apply differential model
        w_t = yaw_vel[t]
        theta_t = state[2, t]
        #todo - double check sinc vs normalized sinc
        #note - applying exact integration, but because steps are so small, is essentially equivalent to euler discretization formula
        integral_term_x = v_t * np.sinc(w_t * T_t / (2*np.pi)) * math.cos(theta_t + w_t * T_t / 2)
        integral_term_y = v_t * np.sinc(w_t * T_t / (2*np.pi)) * math.sin(theta_t + w_t * T_t / 2)
        integral_vec = np.array([integral_term_x, integral_term_y, w_t])
        state[:, t + 1] = state[:, t] + T_t * integral_vec

        #update pose (build body to world)
        pose[0:2,3,t+1] = pose[0:2,3,t] + T_t * integral_vec[0:2] #update p for x and y
        pose[0:3,0:3,t+1]  = build_Rz(state[2, t + 1]) #use new yaw to build rot matrix
        pose[3,3] = 1

    return state[0, :], state[1, :], pose #(x,y,pose)




### LIDAR Helpers (Part 2)

#todo - break this function up more
def perform_LIDAR_odometry(dataset,dataset_index,pointcloud_dataset,pose_odom,num_scans_config,sample_rate):
    """Run scan-matching to estimate odometry from LIDAR data"""
    
    ### Load if previously saved, else recompute
    try:
        LIDAR_Odom = np.load("../data/SavedEstimates/LIDAR_Odom%d.npz" % dataset_index)
    except:
        print("No saved LIDAR Odom data found, recomputing.")
    else:
        print("Loading saved LIDAR Odom data.")
        x_lidar = LIDAR_Odom["x_lidar"]
        y_lidar = LIDAR_Odom["y_lidar"]
        pose_lidar = LIDAR_Odom["pose_lidar"]
        return x_lidar, y_lidar, pose_lidar

    ###Compute LIDAR odometry

    pose_lidar = np.zeros((4,4,dataset.lidar_ranges.shape[1])) #LIDAR-only trajectory
    pose_lidar[:,:,0] = np.eye(4)

    #for each lidar scan, perform scan matching to update pose
        #for scan_index in range(lidar_ranges.shape[1]-1):
    for scan_index in range(num_scans_config//sample_rate):
        if(scan_index%10 == 0):
            print("scan_index:", scan_index)
        #get current and next lidar scan
        pc_a = pointcloud_dataset[:,:,scan_index*sample_rate]#extract_pointcloud(lidar_ranges[:,scan_index*10])
        pc_b = pointcloud_dataset[:,:,scan_index*sample_rate+sample_rate]#extract_pointcloud(lidar_ranges[:,scan_index*10+10])

        # find nearest imu timestamp for t and t+1
        diff = dataset.imu_stamps - dataset.lidar_stamps[scan_index * sample_rate]
        odom_index_a = np.argmin(abs(diff))
        diff = dataset.imu_stamps - dataset.lidar_stamps[(scan_index+1) * sample_rate]
        odom_index_b = np.argmin(abs(diff))

        #find relative pose transform (init with odometry pose estimate)
        pose_wTa = pose_odom[:, :, odom_index_a]
        pose_wTb = pose_odom[:,:,odom_index_b]
        inv_pose_wTa = np.eye(4)
        inv_pose_wTa[0:3,0:3] = pose_wTa[0:3,0:3].transpose() #R transpose
        inv_pose_wTa[0:3,3] = np.matmul((-inv_pose_wTa[0:3,0:3]),pose_wTa[0:3,3]) #-R'p

        relative_pose_init = np.matmul(inv_pose_wTa,pose_wTb) #aTb = inv(wTa)*wTb

        #Note: run_icp(X,Y) produces Pose FROM Y TO X, IE xTy
        relative_pose = run_icp_multiple_inits(pc_a.transpose(), pc_b.transpose(), 2, 10, 0.001, 1,relative_pose_init)

        #multiply pose transform with current wTb pose to get wTb pose for next timestamp
        pose_lidar[:,:,scan_index+1] = np.matmul(pose_lidar[:,:,scan_index],relative_pose) #wTb = wTa*aTb

    #extract position over time
    x_lidar = pose_lidar[0,3,:]
    y_lidar = pose_lidar[1,3,:]

    #save estimate
    LIDAR_Odom = {'x_lidar':x_lidar, 'y_lidar':y_lidar, 'pose_lidar': pose_lidar}
    np.savez("../data/SavedEstimates/LIDAR_Odom%d" % dataset_index, **LIDAR_Odom)
    print("Saved LIDAR Odom data")
    return x_lidar, y_lidar, pose_lidar




### Occupancy Grid Helpers (Part 3)

def point_to_cell(x,y,width,height,grid_res):
    """Translate world coords to occupancy grid coords"""

    col = int((x + width/2)/grid_res)
    row = int((y + height/2)/grid_res)

    #check if point exceeds grid area
    if(row<0 or row>=(height/grid_res) or col<0 or col>=(width/grid_res)):
        row = -1
        col = -1
    return col,row

def update_grid(local_pc,pose,width,height,grid_res, log_odds, occupancy_grid):
    """Update the given occupancy grid for given pointcloud, pose"""
    #pose: wTb (body to world pose transformation)
    #use pose to convert pointcloud to world coords
    new_row = np.ones((1,local_pc.shape[1]))
    local_pc = np.vstack([local_pc,new_row]) #homogenous coord transform
    world_pc = np.matmul(pose,local_pc)
    #bot_pos = pose[0:2,3] #extract x and y
    bot_pos = np.matmul(pose,np.array([0,0,0,1])) #Sw = wTb*Sb
    bot_x, bot_y = point_to_cell(bot_pos[0], bot_pos[1], width, height, grid_res) #get grid coords
    occupancy_grid[bot_y, bot_x] -= 100  # todo - do cleaner assignment of bot location's grid square

    for point_index in range(world_pc.shape[1]):
        point = world_pc[0:2,point_index]

        #get grid coords
        point_x,point_y = point_to_cell(point[0],point[1],width,height,grid_res)

        if(min((bot_x,bot_y,point_x,point_y))<0): #check if any invalid points found
            continue

        #run line alg
        scan_line = bresenham2D(bot_x,bot_y,point_x,point_y)
        if(scan_line.shape[1] <2): #edge/error case - scanned point is bot location
            #print("Error: Bresenham line has length 1")
            continue

        #update intersected points (unoccupied)
        for i in range(1,scan_line.shape[1]-1): #excluding start and end point
            row = int(scan_line[1,i])
            col = int(scan_line[0,i])
            occupancy_grid[row,col] -= log_odds

        occupancy_grid[point_y, point_x] += log_odds #update range point (occupied)

    return occupancy_grid

# TODO - implement load/save code)
def build_occupancy_grid(pose_estimate, pointcloud_dataset, num_scans_config, sample_rate, lidar_trust, width, height, grid_res):
    """
    Given a pointcloud dataset, pose_estimate (robot odometry over time), and occupancy grid
    configs, use the pointcloud data to build out an occupancy grid.
    """
    #set priors
    # – treat all cells as equally likely to be free/occupied
    # - set arbitrary reliability for sensor
    # p_occupied = 0.5
    # o_free = 0.5
    lambda_trust = math.log(lidar_trust/(100-lidar_trust))
    # map_log_odds = zeros() #how to discretize the field?

    occupancy_grid = np.zeros((int(width/grid_res),int(height/grid_res)))


    #iterate through each point cloud scan and update grid:
    for index in range(num_scans_config//sample_rate):#(pose_lidar.shape[2]):
        if(index%10==0):
            print("index: ", index)
        pose_t = pose_estimate[:,:,index]
        pc = pointcloud_dataset[:,:,index*sample_rate]#extract_pointcloud(lidar_ranges[:,index*10])
        occupancy_grid = update_grid(pc, pose_t, width, height, grid_res, lambda_trust, occupancy_grid)

    # limit all occupancy grid vals to -1, 1
    max_indices = np.where(occupancy_grid > 1)
    occupancy_grid[max_indices] = 1
    min_indices = np.where(occupancy_grid < -1)
    occupancy_grid[min_indices] = -1

    return occupancy_grid



### GTSLAM Helpers (Part 4)

def build_pose_graph(pose_lidar, pointcloud_dataset, num_scans_config, sample_rate):
    """Given a pose estimate from LIDAR and a pointcloud dataset, build a pose graph with nodes
    representing robot poses at different timesteps, edges representing the relative transformations
    between consecutive poses, and additional loop closure constraints between certain nodes
    using ICP scan matching"""

    ### Construct factor graph
    graph = gtsam.NonlinearFactorGraph()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.1]) #todo: using deafault, tweak these values
    graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), prior_noise))

    #iterate through every pose and create corresponding node
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.1]) #todo: using default, tweak
    initial_estimate = gtsam.Values()
    initial_estimate.insert( 1, gtsam.Pose2(0, 0, 0))

    #start at index 1 because first node (identity prior) already set
    for index in range(1,num_scans_config//sample_rate):

        if(index%50 == 0):
            print("GTSAM Node Index:", index)

        #get relative pose and extract x, y, yaw
        prev_pose = pose_lidar[:,:,index-1] #get estimated pose of prev node
        current_pose = pose_lidar[:,:,index] #get estimated pose of current node
        rel_pose = np.matmul(invert_pose(prev_pose),current_pose) #get relative pose from current node to previous node
        rot_2d = rel_pose[0:2,0:2] #2d rotation matrix
        yaw = np.arctan2(rot_2d[1,0], rot_2d[0,0]) #extract yaw differnece (todo - double check this)

        #create edge between last and current node, using relative pose
        # - note: add 1 to each index value, because nodes are 1-indexed, not 0-indexed
        # - todo - check that pose from current node to prev node is correct, not vice versa
        graph.add(gtsam.BetweenFactorPose2(index, index+1, gtsam.Pose2(rel_pose[0,3], rel_pose[1,3], yaw),odometry_noise))


        # Add initial estimate (using absolute pose)
        #get absolute pose (body to world)
        rot_2d = current_pose[0:2,0:2]
        yaw = np.arctan2(rot_2d[1,0], rot_2d[0,0]) #double check this
        initial_estimate.insert(index+1, gtsam.Pose2(current_pose[0,3],current_pose[1,3],yaw))


        #add fixed constraint every 10 nodes (between current node and node 10 poses ago)
        interval_rate = 3 #todo - reduce interval rate for quicker runtime
        if((index%interval_rate == 0) and index>=interval_rate):
            #find relative pose using ICP
            #todo - double check doing downsampling correctly here
            pc_a = pointcloud_dataset[:,:,(index-interval_rate)*sample_rate]
            pc_b = pointcloud_dataset[:,:,index*sample_rate]

            #todo - check if need to give ICP better init
            rel_pose = run_icp_multiple_inits(pc_a.transpose(), pc_b.transpose(), 2, 10, 0.001, 1, np.eye(4))
            rot_2d = rel_pose[0:2, 0:2]
            yaw = np.arctan2(rot_2d[1, 0], rot_2d[0, 0])  # double check this
            graph.add(gtsam.BetweenFactorPose2(index-interval_rate+1, index+1, gtsam.Pose2(rel_pose[0,3], rel_pose[1,3], yaw),odometry_noise))

    return graph, initial_estimate

#TODO:
# - improve tuning of pose graph
# - review if combining pose data correctly
def perform_pose_graph_odometry(dataset_index, pose_lidar, pointcloud_dataset, num_scans_config, sample_rate):
    """
    Given odometry estimates from the differential drive motion model, LIDAR, and a pointcloud dataset,
    setup a pose graph with GTSAM and optimize it to get an improved odometry estimate.

    Referenced:
    1.  https://www.roboticsbook.org/S64_driving_perception.html#poseslam)
    2.   https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/Pose2SLAMExample.py
    """

    ### Load if previously saved, else recompute
    try:
        PoseGraph_Odom = np.load("../data/SavedEstimates/PoseGraph_Odom%d.npz" % dataset_index)
    except:
        print("No saved PoseGraph Odom data found, recomputing.")
    else:
        print("Loading saved PoseGraph Odom data.")
        x_gtsam = PoseGraph_Odom["x_gtsam"]
        y_gtsam = PoseGraph_Odom["y_gtsam"]
        pose_gtsam = PoseGraph_Odom["pose_gtsam"]
        return x_gtsam, y_gtsam, pose_gtsam
    
    ### Compute pose graph odometry

    # construct factor graph
    graph, initial_estimate = build_pose_graph(pose_lidar, pointcloud_dataset, num_scans_config, sample_rate)

    #perform optimization
    parameters = gtsam.GaussNewtonParams()
    parameters.setRelativeErrorTol(1e-5)    #set iteration error threshold
    parameters.setMaxIterations(20)
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, parameters)
    result = optimizer.optimize()

    print("Pose Graph Optimization Done")

    #extract(x,y) data from GTSAM estimated pose
    x_gtsam = np.zeros((num_scans_config))
    y_gtsam = np.zeros((num_scans_config))
    pose_gtsam = np.zeros((4,4,num_scans_config))
    pose_gtsam[:,:,0] = np.eye(4)
    for i in range(1, num_scans_config//sample_rate): 
        pose = result.atPose2(i)
        x_gtsam[i] = pose.x() #point[0]
        y_gtsam[i] = pose.y() #point[1]
        pose_gtsam[0,3,i] = x_gtsam[i]
        pose_gtsam[1,3,i] = y_gtsam[i]
        pose_gtsam[0:3,0:3,i]  = build_Rz(pose.theta()) #use new yaw to build rot matrix
        pose_gtsam[3,3] = 1

    #save estimate
    PoseGraph_Odom = {'x_gtsam':x_gtsam, 'y_gtsam':y_gtsam, 'pose_gtsam':pose_gtsam}
    np.savez("../data/SavedEstimates/PoseGraph_Odom%d" % dataset_index, **PoseGraph_Odom)
    print("Saved PoseGraph Odom data")
    return x_gtsam, y_gtsam, pose_gtsam


### Misc Helpers
def invert_pose(pose):
    """Invert 4x4 pose matrix (given A to B pose transform, return B to A pose transform)"""
    inv_pose = np.eye(4)
    inv_pose[0:3, 0:3] = pose[0:3, 0:3].transpose()  # R transpose
    inv_pose[0:3, 3] = np.matmul((-inv_pose[0:3, 0:3]), pose[0:3, 3])  # -R'p
    return inv_pose


# todo - consolidate overlap between plot_trajectories() and plot_SLAM() code
def plot_trajectories(trajectory_estimates, trajectory_estimate_labels, plot_title):
    """Given a set of different trajectory estimates and corresponding labels, plot each trajectory"""

    plt.axis('equal')
    plt.title(plot_title)
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters')
    for x_estimate,y_estimate in trajectory_estimates:
        plt.plot(x_estimate, y_estimate)
    plt.legend(trajectory_estimate_labels)
    plt.show()


def plot_SLAM(trajectory_estimates, trajectory_estimate_labels, occupancy_grid, width, grid_res):
    """Plot trajectory estimate(s) overlaid on occupancy map"""

    # plot occupancy map
    cmap = mpl.colors.ListedColormap(['white', 'grey', 'black'])
    plt.title("Occupancy Grid:")
    # plt.xlabel('x (meters)')
    # plt.ylabel('y (meters')
    img = plt.imshow(occupancy_grid, cmap=cmap, origin='lower')

    # overlay each trajectory
    for x_estimate,y_estimate in trajectory_estimates:
        converted_x_estimate = (x_estimate[:] + width / 2) / grid_res
        converted_y_estimate = (y_estimate[:] + width / 2) / grid_res
        plt.plot(converted_x_estimate, converted_y_estimate)
    
    # plt setup
    plt.colorbar(img)
    plt.ylabel('x (0.1 meters)')
    plt.ylabel('y (0.1 meters')
    plt.legend(trajectory_estimate_labels)
    plt.show()

