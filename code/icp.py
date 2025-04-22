"""
LIDAR SLAM Project - Winter 2024
Author: Aram Chemishkian (achemish@gmail.com)
icp.py: 
    This file implements the ICP (Iterative Closest Point) algorithm to find the pose transformation
    between two point clouds with unknown point associations.
"""

import math

import numpy as np
from sklearn.neighbors import KDTree


def apply_Kabsch(pc_M, pc_Z):
    """
    Given two pointclouds with known associations between points, find the
    pose transformation between the pointclouds.

    Args:
        pc_M (ndarray): pointcloud
        pc_Z (ndarray): poincloud of same environment/object as M from a different angle,
                        with points ordered to corresponding to same features as in M

    Returns:
        p(ndarray): translation from pointcloud Z to M
        R(ndarray): rotation from pointcloud Z to M
    """
    assert (isinstance(pc_M, np.ndarray))
    assert (isinstance(pc_Z, np.ndarray))
    assert (pc_M.shape[0] == 3)
    assert (pc_Z.shape[0] == 3)
    assert (pc_M.shape[1] == pc_Z.shape[1])

    num_points = pc_M.shape[1]

    # center point clouds
    m_sum = np.sum(pc_M, 1)
    m_bar = m_sum / num_points  # todo - verify if I should divide by num points
    m_centered = pc_M - m_bar.reshape(-1, 1)
    z_sum = np.sum(pc_Z, 1)
    z_bar = z_sum / num_points  # todo - verify if I should divide by num points
    z_centered = pc_Z - z_bar.reshape(-1, 1)

    # todo - remove once verify code works
    assert (m_sum.shape[0] == 3)
    assert (m_centered.shape == pc_M.shape)
    assert (z_centered.shape == pc_Z.shape)

    # formulate and solve Wahba's problem
    Q = np.zeros((3,3))
    sum_mat = np.matmul(m_centered, z_centered.transpose())
    assert (sum_mat.shape == (3, 3))
    Q = sum_mat

    # apply Wahba closed form solution
    U, S, Vh = np.linalg.svd(Q)
    S = np.matmul(np.eye(3), S)
    det_term = np.linalg.det(np.matmul(U, Vh))
    M = np.eye(3)
    M[2, 2] = det_term  # todo - verify that this op is applied correctly, changed code from jax implementation.
    R = np.matmul(np.matmul(U, M), Vh)

    # use R to compute translation
    p = m_bar - np.matmul(R, z_bar)

    return p, R

def run_icp_single_init(pc_M, pc_Z, tree_M, p_init, R_init, min_iter, max_iter, err_threshold, debug_print_flag = False):
    """
    Run iterations of ICP algorithm from a given initialization until below error threshold or
    maximum iterations reached.

    Steps:
    1. Using current relative transform estimate, find the best possible associations
        between pointcloud M and Z, and reorder M to match Z
    2. Compute avg point association error and check exit criteria
    3. Using our guessed point associations, apply the closed-form Kabsch solution to
       Wahba's problem (pose transformation between two point clouds with known point associations)
       to get an updated relative transform estimate.

    Args:
        pc_M (ndarray): pointcloud from single scan
        pc_Z (ndarray): pointcloud from single scan (capturing same environment as M, but point
                        associations unknown)
        tree_M (KDTree): KDtree of pc_M points for efficient nearest neighbor searches
        p_init (ndarray): Initial translation estimate.
        R_init (ndarray): Initial rotation estimate.
        min_iter (int): minimal number of iterations to run
        max_iter (int): maximum number of iterations to run
        err_threshold (float): error threshold under which to end iteration
        debug_print_flag (bool, optional): Set true for verbose output. Defaults to False.

    Returns:
        p(ndarray): estimated translation from pointcloud Z to M
        R(ndarray): estimated rotation from pointcloud Z to M
        avg_error(float): avg error between estimated associated points
    """
    #icp setup
    R = R_init
    p = p_init
    iter_index = 0
    ordered_M = np.zeros((3, pc_Z.shape[1]))

    # perform ICP iteration until exit condition met
    while (True):

        # debug info
        if(debug_print_flag):
            print("iter_index: ", iter_index)
            # print("R:")
            # print(R)
            # print("p:")
            # print(p)

        # Step 1: find point correspondences
        # - for each i point in M, find closest j point in Z
        transformed_Z = np.matmul(R, pc_Z) + p.reshape(-1, 1)  # apply Z to M transform outside of loop for efficiency
        for j in range(pc_Z.shape[1]):
            z_j = transformed_Z[:, j].reshape(1, -1)

            # find the closest point to z_j in M
            dist, ind = tree_M.query(z_j, 1)
            ordered_M[:, j] = pc_M[:, ind[0, 0]]  # save matched point

        #todo - find way to get all nearest neighbors at once
        # distances ,indices = tree_M.query(translated_Z, 1)
        # ordered_M[:,:] = pc_M[:, indices]  # save matched point

        # Step 2: Check exit conditions:

        # compute avg error (for exit criteria & comparing icp runs)
        point_diff = abs(transformed_Z - ordered_M)
        avg_error = np.sum(point_diff)/point_diff.shape[1] #divide by #points, to get avg point error
        if(debug_print_flag):
            print("avg error: ", avg_error)

        # check exit criteria
        if (iter_index >= max_iter or (iter_index >= min_iter and avg_error <= err_threshold)):
            break
        iter_index += 1

        # Step 3: apply Kabsch for our estimated point associations
        p, R = apply_Kabsch(ordered_M, pc_Z)

    return p, R, avg_error

def build_Rz(yaw):
    """Given yaw angle (rads), return a corresponding 3x3 3D rotation matrix"""
    return [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0,0,1]]

def run_icp_multiple_inits(pc_M, pc_Z, min_iter = 10, max_iter = 30, err_threshold = 0.0, num_inits=4,
            pose_init=np.eye(4), debug_flag = False):
    """
    Given two pointclouds of the same environment with unknown point associations, run ICP
    multiple times with different intializations to estimate the relative transform between the
    pointclouds.

    Args:
        pc_M (ndarray): pointcloud from single scan
        pc_Z (ndarray): pointcloud from single scan (capturing same environment as M, but point
                        associations unknown)
        min_iter (int, optional): Min number of iterations to run before ending. Defaults to 10.
        max_iter (int, optional): Max number of iterations to run before ending. Defaults to 30.
        err_threshold (float, optional): Avg error below which to end iteration. Defaults to 0.0.
        num_inits (int, optional): Number of different inits to run ICP algorithm from. Defaults to 4.
        pose_init (ndarray, optional): Relative pose to use as ICP algorithm init. Defaults to np.eye(4).
        debug_flag (bool, optional): Set true for verbose output. Defaults to False.

    Returns:
        pose (ndarray): relative transform from pointcloud Z to M
    """
    #TODO - check and re-enable different initializations, disabled it previously for some reason

    assert (isinstance(pc_M, np.ndarray))
    assert (isinstance(pc_Z, np.ndarray))
    assert (pc_M.shape[1] == 3)
    assert (pc_Z.shape[1] == 3)
    # assert (pc_M.shape[0] == pc_Z.shape[0])
    #num_points = pc_M.shape[0]

    # fix matrix dimensions
    pc_M = np.transpose(pc_M)
    pc_Z = np.transpose(pc_Z)

    #downsample
    pc_M = pc_M[:,::2]
    pc_Z = pc_Z[:,::2]

    # ICP setup
    # R_init = np.eye(3)
    yaw = math.pi/4
    # R_z = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0,0,1]]
    p_init = pose_init[0:3,3]
    R_init = pose_init[0:3,0:3]

    #p_init = np.zeros((3, 1))
    tree_M = KDTree(pc_M.transpose()) #setup KDtree for efficient nearest point computation
                          #todo - adjust leaf size?
    best_R = np.eye(3)
    best_p = np.zeros((3,1))
    best_init_yaw = 0
    min_error = 1000 #todo - replace placeholder
    for x in range(num_inits):
        # init_yaw = x*2*math.pi/num_inits#x*math.pi/2
        # R_init = build_Rz(init_yaw) #iterate init R from 0 to 270 degree rotation, in 90 degree steps
        p, R, avg_error = run_icp_single_init(pc_M, pc_Z, tree_M, p_init, R_init, min_iter, max_iter, err_threshold, debug_flag)
        if(debug_flag):
            #print("Init Yaw: ", init_yaw)
            print("Error: ", avg_error)
        if(avg_error < min_error):
            min_error = avg_error
            best_R = R
            best_p = p
            #best_init_yaw = init_yaw

    # compose pose
    pose = np.eye(4)
    pose[0:3, 0:3] = best_R
    pose[0:3, 3] = best_p
    if(debug_flag):
        print("best init yaw:", best_init_yaw)
        print("min error:", min_error)
        print("final pose", pose)

    return pose