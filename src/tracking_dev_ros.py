#!/usr/bin/env python

import matplotlib.pyplot as plt
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg

import struct
import time
import cv2
import numpy as np
import math

import time
import pickle as pkl

import message_filters
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from scipy import ndimage

def pt2pt_dis_sq(pt1, pt2):
    return np.sum(np.square(pt1 - pt2))

def pt2pt_dis(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))

def register(pts, M, mu=0, max_iter=50):

    # initial guess
    X = pts.copy()
    Y = np.vstack((np.arange(0, 0.1, (0.1/M)), np.zeros(M), np.zeros(M))).T
    if len(pts[0]) == 2:
        Y = np.vstack((np.arange(0, 0.1, (0.1/M)), np.zeros(M))).T
    s = 1
    N = len(pts)
    D = len(pts[0])

    def get_estimates (Y, s):

        # construct the P matrix
        P = np.sum((X[None, :, :] - Y[:, None, :]) ** 2, axis=2)

        c = (2 * np.pi * s) ** (D / 2)
        c = c * mu / (1 - mu)
        c = c * M / N

        P = np.exp(-P / (2 * s))
        den = np.sum(P, axis=0)
        den = np.tile(den, (M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c

        P = np.divide(P, den)  # P is M*N
        Pt1 = np.sum(P, axis=0)  # equivalent to summing from 0 to M (results in N terms)
        P1 = np.sum(P, axis=1)  # equivalent to summing from 0 to N (results in M terms)
        Np = np.sum(P1)
        PX = np.matmul(P, X)

        # get new Y
        P1_expanded = np.full((D, M), P1).T
        new_Y = PX / P1_expanded

        # get new sigma2
        Y_N_arr = np.full((N, M, 3), Y)
        Y_N_arr = np.swapaxes(Y_N_arr, 0, 1)
        X_M_arr = np.full((M, N, 3), X)
        diff = Y_N_arr - X_M_arr
        diff = np.square(diff)
        diff = np.sum(diff, 2)
        new_s = np.sum(np.sum(P*diff, axis=1), axis=0) / (Np*D)

        return new_Y, new_s

    prev_Y, prev_s = Y, s
    new_Y, new_s = get_estimates(prev_Y, prev_s)
    # it = 0
    tol = 0.0
    
    for it in range (max_iter):
        prev_Y, prev_s = new_Y, new_s
        new_Y, new_s = get_estimates(prev_Y, prev_s)

    # print(repr(new_x), new_s)
    return new_Y, new_s

# assuming Y is sorted
# k -- going left for k indices, going right for k indices. a total of 2k neighbors.
def get_nearest_indices (k, Y, idx):
    if idx - k < 0:
        # use more neighbors from the other side?
        indices_arr = np.append(np.arange(0, idx, 1), np.arange(idx+1, idx+k+1+np.abs(idx-k)))
        # indices_arr = np.append(np.arange(0, idx, 1), np.arange(idx+1, idx+k+1))
        return indices_arr
    elif idx + k >= len(Y):
        last_index = len(Y) - 1
        # use more neighbots from the other side?
        indices_arr = np.append(np.arange(idx-k-(idx+k-last_index), idx, 1), np.arange(idx+1, last_index+1, 1))
        # indices_arr = np.append(np.arange(idx-k, idx, 1), np.arange(idx+1, last_index+1, 1))
        return indices_arr
    else:
        indices_arr = np.append(np.arange(idx-k, idx, 1), np.arange(idx+1, idx+k+1, 1))
        return indices_arr

def calc_LLE_weights (k, X):
    W = np.zeros((len(X), len(X)))
    for i in range (0, len(X)):
        indices = get_nearest_indices(int(k/2), X, i)
        xi, Xi = X[i], X[indices, :]
        component = np.full((len(Xi), len(xi)), xi).T - Xi.T
        Gi = np.matmul(component.T, component)
        # Gi might be singular when k is large
        try:
            Gi_inv = np.linalg.inv(Gi)
        except:
            epsilon = 0.00001
            Gi_inv = np.linalg.inv(Gi + epsilon*np.identity(len(Gi)))
        wi = np.matmul(Gi_inv, np.ones((len(Xi), 1))) / np.matmul(np.matmul(np.ones(len(Xi),), Gi_inv), np.ones((len(Xi), 1)))
        W[i, indices] = np.squeeze(wi.T)

    return W

def indices_array(n):
    r = np.arange(n)
    out = np.empty((n,n,2),dtype=int)
    out[:,:,0] = r[:,None]
    out[:,:,1] = r
    return out

def ecpd_lle (X,                           # input point cloud
              Y_0,                         # input nodes
              beta,                        # MCT kernel strength
              alpha,                       # MCT overall strength
              gamma,                       # LLE strength
              mu,                          # noise
              max_iter,                    # how many iterations EM will run
              tol,                         # when to terminate the optimization process
              include_lle = True, 
              use_geodesic = False, 
              use_prev_sigma2 = False, 
              sigma2_0 = None,              # initial variance
              use_ecpd = False, 
              correspondence_priors = None,
              omega = None,                 # ecpd strength. DO NOT go lower than 1e-6
              kernel = 'Gaussian',          # Gaussian, Laplacian, 1st order, 2nd order
              occluded_nodes = None):

    if correspondence_priors is not None:
        additional_pc = correspondence_priors[:, 1:4]
        X = np.vstack((additional_pc, X))

    # define params
    M = len(Y_0)
    N = len(X)
    D = 3

    # initialization
    # faster G calculation
    diff = Y_0[:, None, :] - Y_0[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)

    converted_node_dis = []
    if not use_geodesic:
        if kernel == 'Gaussian':
            G = np.exp(-diff / (2 * beta**2))
        elif kernel == 'Laplacian':
            G = np.exp(- np.sqrt(diff) / (2 * beta**2))
        elif kernel == '1st order':
            G = 1/(2*beta)**2 * np.exp(-np.sqrt(2)*np.sqrt(diff)/beta) * (np.sqrt(2)*np.sqrt(diff) + beta)
        elif kernel == '2nd order':
            G = 27 * 1/(72*beta**3) * np.exp(-math.sqrt(3)*np.sqrt(diff)/beta) * (np.sqrt(3)*beta**2 + 3*beta*np.sqrt(diff) + np.sqrt(3)*diff)
        else: # default gaussian
            G = np.exp(-diff / (2 * beta**2))
    else:
        seg_dis = np.sqrt(np.sum(np.square(np.diff(Y_0, axis=0)), axis=1))
        converted_node_coord = []
        last_pt = 0
        converted_node_coord.append(last_pt)
        for i in range (1, M):
            last_pt += seg_dis[i-1]
            converted_node_coord.append(last_pt)
        converted_node_coord = np.array(converted_node_coord)
        converted_node_dis = np.abs(converted_node_coord[None, :] - converted_node_coord[:, None])
        converted_node_dis_sq = np.square(converted_node_dis)

        if kernel == 'Gaussian':
            G = np.exp(-converted_node_dis_sq / (2 * beta**2))
        elif kernel == 'Laplacian':
            G = np.exp(-converted_node_dis / (2 * beta**2))
        elif kernel == '1st order':
            G = 1/(4*beta**2) * np.exp(-np.sqrt(2)*converted_node_dis/beta) * (np.sqrt(2)*converted_node_dis + beta)
        elif kernel == '2nd order':
            G = 27 * 1/(72*beta**3) * np.exp(-math.sqrt(3)*converted_node_dis/beta) * (np.sqrt(3)*beta**2 + 3*beta*converted_node_dis + np.sqrt(3)*converted_node_dis_sq)
        else:
            G = np.exp(-converted_node_dis_sq / (2 * beta**2))
    
    Y = Y_0.copy()

    # initialize sigma2
    if not use_prev_sigma2:
        (N, D) = X.shape
        (M, _) = Y.shape
        diff = X[None, :, :] - Y[:, None, :]
        err = diff ** 2
        sigma2 = np.sum(err) / (D * M * N)
    else:
        sigma2 = sigma2_0

    # get the LLE matrix
    cur_time = time.time()
    L = calc_LLE_weights(6, Y_0)
    H = np.matmul((np.identity(M) - L).T, np.identity(M) - L)
    
    # loop until convergence or max_iter reached
    for it in range (0, max_iter):

        # faster P computation
        pts_dis_sq = np.sum((X[None, :, :] - Y[:, None, :]) ** 2, axis=2)
        c = (2 * np.pi * sigma2) ** (D / 2)
        c = c * mu / (1 - mu)
        c = c * M / N
        P = np.exp(-pts_dis_sq / (2 * sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c
        P = np.divide(P, den)
        max_p_nodes = np.argmax(P, axis=0)

        if use_geodesic:
            potential_2nd_max_p_nodes_1 = max_p_nodes - 1
            potential_2nd_max_p_nodes_2 = max_p_nodes + 1
            potential_2nd_max_p_nodes_1 = np.where(potential_2nd_max_p_nodes_1 < 0, 1, potential_2nd_max_p_nodes_1)
            potential_2nd_max_p_nodes_2 = np.where(potential_2nd_max_p_nodes_2 > M-1, M-2, potential_2nd_max_p_nodes_2)
            potential_2nd_max_p_nodes_1_select = np.vstack((np.arange(0, N), potential_2nd_max_p_nodes_1)).T
            potential_2nd_max_p_nodes_2_select = np.vstack((np.arange(0, N), potential_2nd_max_p_nodes_2)).T
            potential_2nd_max_p_1 = P.T[tuple(map(tuple, potential_2nd_max_p_nodes_1_select.T))]
            potential_2nd_max_p_2 = P.T[tuple(map(tuple, potential_2nd_max_p_nodes_2_select.T))]
            next_max_p_nodes = np.where(potential_2nd_max_p_1 > potential_2nd_max_p_2, potential_2nd_max_p_nodes_1, potential_2nd_max_p_nodes_2)
            node_indices_diff = max_p_nodes - next_max_p_nodes
            max_node_smaller_index = np.arange(0, N)[node_indices_diff < 0]
            max_node_larger_index = np.arange(0, N)[node_indices_diff > 0]
            dis_to_max_p_nodes = np.sqrt(np.sum(np.square(Y[max_p_nodes]-X), axis=1))
            dis_to_2nd_largest_p_nodes = np.sqrt(np.sum(np.square(Y[next_max_p_nodes]-X), axis=1))
            converted_P = np.zeros((M, N)).T

            for idx in max_node_smaller_index:
                converted_P[idx, 0:max_p_nodes[idx]+1] = converted_node_dis[max_p_nodes[idx], 0:max_p_nodes[idx]+1] + dis_to_max_p_nodes[idx]
                converted_P[idx, next_max_p_nodes[idx]:M] = converted_node_dis[next_max_p_nodes[idx], next_max_p_nodes[idx]:M] + dis_to_2nd_largest_p_nodes[idx]

            for idx in max_node_larger_index:
                converted_P[idx, 0:next_max_p_nodes[idx]+1] = converted_node_dis[next_max_p_nodes[idx], 0:next_max_p_nodes[idx]+1] + dis_to_2nd_largest_p_nodes[idx]
                converted_P[idx, max_p_nodes[idx]:M] = converted_node_dis[max_p_nodes[idx], max_p_nodes[idx]:M] + dis_to_max_p_nodes[idx]

            converted_P = converted_P.T

            P = np.exp(-np.square(converted_P) / (2 * sigma2))
            den = np.sum(P, axis=0)
            den = np.tile(den, (M, 1))
            den[den == 0] = np.finfo(float).eps
            c = (2 * np.pi * sigma2) ** (D / 2)
            c = c * mu / (1 - mu)
            c = c * M / N
            den += c

            P = np.divide(P, den)

        if occluded_nodes is not None:
            P[occluded_nodes] = 0

        Pt1 = np.sum(P, axis=0)
        P1 = np.sum(P, axis=1)
        Np = np.sum(P1)
        PX = np.matmul(P, X)
    
        # M step
        if include_lle:
            if use_ecpd:
                P_tilde = np.zeros((M, N))
                # correspondence priors: index, x, y, z
                for i in range (len(correspondence_priors)):
                    index = correspondence_priors[i, 0]
                    P_tilde[int(index), i] = 1

                P_tilde_1 = np.sum(P_tilde, axis=1)
                P_tilde_X = np.matmul(P_tilde, X)

                A_matrix = np.matmul(np.diag(P1), G) + alpha * sigma2 * np.identity(M) + sigma2 * gamma * np.matmul(H, G) + sigma2 / omega * np.matmul(np.diag(P_tilde_1), G)
                B_matrix = PX - np.matmul(np.diag(P1) + sigma2*gamma*H, Y_0) + sigma2 / omega * (P_tilde_X - np.matmul(np.diag(P_tilde_1) + sigma2*gamma*H, Y_0))
            else:
                A_matrix = np.matmul(np.diag(P1), G) + alpha * sigma2 * np.identity(M) + sigma2 * gamma * np.matmul(H, G)
                B_matrix = PX - np.matmul(np.diag(P1) + sigma2*gamma*H, Y_0)
        else:
            if use_ecpd:
                P_tilde = np.zeros((M, N))
                pt_node_correspondence = np.argmax(P, axis=0)
                
                for node_num in range (0, M):
                    node_num_pts_indices = np.where(pt_node_correspondence == node_num)
                    P_tilde[node_num, node_num_pts_indices] = 1

                P_tilde_1 = np.sum(P_tilde, axis=1)
                P_tilde_X = np.matmul(P_tilde, X)

                A_matrix = np.matmul(np.diag(P1), G) + alpha * sigma2 * np.identity(M) + sigma2 / omega * np.matmul(np.diag(P_tilde_1), G)
                B_matrix = PX - np.matmul(np.diag(P1), Y_0) + sigma2 / omega * (P_tilde_X - np.matmul(np.diag(P_tilde_1), Y_0))
            else:
                A_matrix = np.matmul(np.diag(P1), G) + alpha * sigma2 * np.identity(M)
                B_matrix = PX - np.matmul(np.diag(P1), Y_0)

        W = np.linalg.solve(A_matrix, B_matrix)

        T = Y_0 + np.matmul(G, W)
        trXtdPt1X = np.trace(np.matmul(np.matmul(X.T, np.diag(Pt1)), X))
        trPXtT = np.trace(np.matmul(PX.T, T))
        trTtdP1T = np.trace(np.matmul(np.matmul(T.T, np.diag(P1)), T))

        sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D)

        # update Y
        if pt2pt_dis_sq(Y, Y_0 + np.matmul(G, W)) < tol:
            Y = Y_0 + np.matmul(G, W)
            break
        else:
            Y = Y_0 + np.matmul(G, W)

    return Y, sigma2

def pre_process (X, Y_0, geodesic_coord, total_len):
    guide_nodes, _ = ecpd_lle(X, Y_0, 0.5, 1, 1, 0.1, 30, 0.00001, True, True, kernel = 'Gaussian')

    # determine which head node is occluded, if any
    head_visible = False
    tail_visible = False

    if pt2pt_dis(guide_nodes[0], Y_0[0]) < 0.01:
        head_visible = True
    if pt2pt_dis(guide_nodes[-1], Y_0[-1]) < 0.01:
        tail_visible = True

    cur_total_len = np.sum(np.sqrt(np.sum(np.square(np.diff(guide_nodes, axis=0)), axis=1)))

    print('tail displacement = ', pt2pt_dis(guide_nodes[-1], Y_0[-1]))
    print('head displacement = ', pt2pt_dis(guide_nodes[0], Y_0[0]))
    print('length difference = ', abs(cur_total_len - total_len))

    # visible_dist = np.sum(np.sqrt(np.sum(np.square(np.diff(guide_nodes, axis=0)), axis=1)))
    last_visible_index = None
    correspondence_priors = None
    occluded_nodes = None

    if (head_visible and tail_visible) or abs(cur_total_len - total_len) < 0.005:
        print('head visible and tail visible or the same len')
        correspondence_priors = []
        correspondence_priors.append(np.append(np.array([0]), guide_nodes[0]))
        correspondence_priors.append(np.append(np.array([len(guide_nodes)-1]), guide_nodes[-1]))
    
    elif head_visible and not tail_visible:
        print('head visible')
        correspondence_priors = []
        total_dist_Y_0 = 0
        total_dist_guide_nodes = 0
        it_gn = 0
        correspondence_priors.append(np.append(np.array([0]), guide_nodes[0]))
        for it_y0 in range (0, len(guide_nodes)-1):
            total_dist_Y_0 += pt2pt_dis(geodesic_coord[it_y0], geodesic_coord[it_y0+1])
            # step guide nodes dist until greater than current y0 total dist
            while total_dist_guide_nodes < total_dist_Y_0:
                total_dist_guide_nodes += pt2pt_dis(guide_nodes[it_gn], guide_nodes[it_gn+1])
                if total_dist_guide_nodes >= total_dist_Y_0:
                    total_dist_guide_nodes -= pt2pt_dis(guide_nodes[it_gn], guide_nodes[it_gn+1])
                    new_y0_coord = guide_nodes[it_gn] + (total_dist_Y_0 - total_dist_guide_nodes)/pt2pt_dis(guide_nodes[it_gn], guide_nodes[it_gn + 1])*(guide_nodes[it_gn + 1] - guide_nodes[it_gn])
                    correspondence_priors.append(np.append(np.array([it_y0+1]), new_y0_coord))
                    break
                # if at the end of guide nodes
                if it_gn == len(guide_nodes) - 2:
                    last_visible_index = it_y0
                    break
                it_gn += 1
            
            if last_visible_index is not None:
                break
        if last_visible_index is None:
            last_visible_index = len(Y_0) - 1
        
        occluded_nodes = np.arange(last_visible_index+1, len(Y_0), 1)
        print(last_visible_index)

    elif tail_visible and not head_visible and not abs(cur_total_len - total_len) < 0.01:
        print('tail visible')
        correspondence_priors = []
        total_dist_Y_0 = 0
        total_dist_guide_nodes = 0
        it_gn = len(guide_nodes) - 1
        correspondence_priors.append(np.append(np.array([len(guide_nodes)-1]), guide_nodes[-1]))
        for it_y0 in range (len(guide_nodes)-1, 0, -1):
            total_dist_Y_0 += pt2pt_dis(geodesic_coord[it_y0], geodesic_coord[it_y0-1])
            # step guide nodes dist until greater than current y0 total dist
            while total_dist_guide_nodes < total_dist_Y_0:
                total_dist_guide_nodes += pt2pt_dis(guide_nodes[it_gn], guide_nodes[it_gn-1])
                if total_dist_guide_nodes >= total_dist_Y_0:
                    total_dist_guide_nodes -= pt2pt_dis(guide_nodes[it_gn], guide_nodes[it_gn-1])
                    new_y0_coord = guide_nodes[it_gn] + (total_dist_Y_0 - total_dist_guide_nodes)/pt2pt_dis(guide_nodes[it_gn], guide_nodes[it_gn - 1])*(guide_nodes[it_gn - 1] - guide_nodes[it_gn])
                    correspondence_priors.append(np.append(np.array([it_y0-1]), new_y0_coord))
                    break
                # if at the end of guide nodes
                if it_gn == 1:
                    last_visible_index = it_y0
                    break
                it_gn -= 1
            
            if last_visible_index is not None:
                break
        if last_visible_index is None:
            last_visible_index = 0
        
        occluded_nodes = np.arange(0, last_visible_index, 1)
    
    print(np.array(correspondence_priors)[-1, 0])

    return guide_nodes, np.array(correspondence_priors), occluded_nodes

def tracking_step (X, Y_0, sigma2_0, geodesic_coord, total_len):
    guide_nodes, correspondence_priors, occluded_nodes = pre_process(X, Y_0, geodesic_coord, total_len)
    Y, sigma2 = ecpd_lle(X, Y_0, 2, 1, 1, 0.1, 30, 0.00001, True, True, True, sigma2_0, True, correspondence_priors, 0.00001, '2nd order', occluded_nodes)

    return correspondence_priors[:, 1:4], Y, sigma2

def find_closest (pt, arr):
    closest = arr[0].copy()
    min_dis = np.sqrt((pt[0] - closest[0])**2 + (pt[1] - closest[1])**2 + (pt[2] - closest[2])**2)
    idx = 0

    for i in range (0, len(arr)):
        cur_pt = arr[i].copy()
        cur_dis = np.sqrt((pt[0] - cur_pt[0])**2 + (pt[1] - cur_pt[1])**2 + (pt[2] - cur_pt[2])**2)
        if cur_dis < min_dis:
            min_dis = cur_dis
            closest = arr[i].copy()
            idx = i
    
    return closest, idx

def find_opposite_closest (pt, arr, direction_pt):
    arr_copy = arr.copy()
    opposite_closest_found = False
    opposite_closest = pt.copy()  # will get overwritten

    while (not opposite_closest_found) and (len(arr_copy) != 0):
        cur_closest, cur_index = find_closest (pt, arr_copy)
        arr_copy.pop (cur_index)

        vec1 = np.array(cur_closest) - np.array(pt)
        vec2 = np.array(direction_pt) - np.array(pt)

        # threshold: 0.07m
        if (np.dot (vec1, vec2) < 0) and (pt2pt_dis_sq(np.array(cur_closest), np.array(pt)) < 0.07**2):
            opposite_closest_found = True
            opposite_closest = cur_closest.copy()
            break
    
    return opposite_closest, opposite_closest_found

def sort_pts (pts_orig):

    start_idx = 5

    pts = pts_orig.copy()
    starting_pt = pts[start_idx].copy()
    pts.pop(start_idx)
    # starting point will be the current first point in the new list
    sorted_pts = []
    sorted_pts.append(starting_pt)

    # get the first closest point
    closest_1, min_idx = find_closest (starting_pt, pts)
    sorted_pts.append(closest_1)
    pts.pop(min_idx)

    # get the second closest point
    closest_2, found = find_opposite_closest(starting_pt, pts, closest_1)
    true_start = False
    if not found:
        # closest 1 is true start
        true_start = True
    # closest_2 is not popped from pts

    # move through the rest of pts to build the sorted pt list
    # if true_start:
    #   can proceed until pts is empty
    # if !true_start:
    #   start in the direction of closest_1, the list would build until one end is reached. 
    #   in that case the next closest point would be closest_2. starting that point, all 
    #   newly added points to sorted_pts should be inserted at the front
    while len(pts) != 0:
        cur_target = sorted_pts[-1]
        cur_direction = sorted_pts[-2]
        cur_closest, found = find_opposite_closest(cur_target, pts, cur_direction)

        if not found:
            print ("not found!")
            break

        sorted_pts.append(cur_closest)
        pts.remove (cur_closest)

    # begin the second loop that inserts new points at front
    if not true_start:
        # first insert closest_2 at front and pop it from pts
        sorted_pts.insert(0, closest_2)
        pts.remove(closest_2)

        while len(pts) != 0:
            cur_target = sorted_pts[0]
            cur_direction = sorted_pts[1]
            cur_closest, found = find_opposite_closest(cur_target, pts, cur_direction)

            if not found:
                print ("not found!")
                break

            sorted_pts.insert(0, cur_closest)
            pts.remove(cur_closest)

    return sorted_pts

saved = False
initialized = False
init_nodes = []
nodes = []
sigma2 = 0
cur_time = time.time()
total_len = 0
geodesic_coord = []
def callback (rgb, depth, pc):
    global saved
    global initialized
    global init_nodes
    global nodes
    global sigma2
    global cur_time
    global total_len
    global geodesic_coord

    proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                            [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                            [             0.0,              0.0,               1.0, 0.0]])

    # process rgb image
    cur_image = ros_numpy.numpify(rgb)
    # cur_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_RGB2HSV)

    # # test
    # cv2.imshow('img', cur_image)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

    # process depth image
    cur_depth = ros_numpy.numpify(depth)

    # process point cloud
    pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc)
    cur_pc = ros_numpy.point_cloud2.get_xyz_points(pc_data)
    cur_pc = cur_pc.reshape((720, 1280, 3))

    # color thresholding
    lower = (90, 100, 100)
    upper = (120, 255, 255)
    mask = cv2.inRange(hsv_image, lower, upper)
    bmask = mask.copy() # for checking visibility, max = 255
    mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    # print('mask shape = ', np.shape(mask))

    # publish mask
    mask_img_msg = ros_numpy.msgify(Image, mask, 'rgb8')
    mask_img_pub.publish(mask_img_msg)

    mask = (mask/255).astype(int)

    filtered_pc = cur_pc*mask
    filtered_pc = filtered_pc[((filtered_pc[:, :, 0] != 0) | (filtered_pc[:, :, 1] != 0) | (filtered_pc[:, :, 2] != 0))]
    filtered_pc = filtered_pc[filtered_pc[:, 2] < 0.705]
    filtered_pc = filtered_pc[filtered_pc[:, 2] > 0.4]

    # # save points
    # if not saved:
    #     username = 'ablcts18'
    #     folder = 'tracking/'
    #     f = open("/home/" + username + "/Research/" + folder + "ros_pc.json", 'wb')
    #     pkl.dump(filtered_pc, f)
    #     f.close()
    #     saved = True

    # downsample to 2.5%
    # filtered_pc = filtered_pc[::int(1/0.1)]

    # downsample with open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_pc)
    downpcd = pcd.voxel_down_sample(voxel_size=0.005)
    filtered_pc = np.asarray(downpcd.points)

    # add color
    pc_rgba = struct.unpack('I', struct.pack('BBBB', 255, 40, 40, 255))[0]
    pc_rgba_arr = np.full((len(filtered_pc), 1), pc_rgba)
    filtered_pc_colored = np.hstack((filtered_pc, pc_rgba_arr)).astype('O')
    filtered_pc_colored[:, 3] = filtered_pc_colored[:, 3].astype(int)

    # filtered_pc = filtered_pc.reshape((len(filtered_pc)*len(filtered_pc[0]), 3))
    header.stamp = rospy.Time.now()
    converted_points = pcl2.create_cloud(header, fields, filtered_pc_colored)
    pc_pub.publish(converted_points)

    # register nodes
    if not initialized:
        init_nodes, sigma2 = register(filtered_pc, 10, mu=0.05, max_iter=100)
        init_nodes = np.array(sort_pts(init_nodes.tolist()))

        # compute preset coord and total len. one time action
        seg_dis = np.sqrt(np.sum(np.square(np.diff(init_nodes, axis=0)), axis=1))
        geodesic_coord = []
        last_pt = 0
        geodesic_coord.append(last_pt)
        for i in range (1, len(init_nodes)):
            last_pt += seg_dis[i-1]
            geodesic_coord.append(last_pt)
        geodesic_coord = np.array(geodesic_coord)
        total_len = np.sum(np.sqrt(np.sum(np.square(np.diff(init_nodes, axis=0)), axis=1)))

        initialized = True
        # header.stamp = rospy.Time.now()
        # converted_init_nodes = pcl2.create_cloud(header, fields, init_nodes)
        # init_nodes_pub.publish(converted_init_nodes)

    # cpd
    if initialized:
        # determined which nodes are occluded from mask information
        mask_dis_threshold = 30
        # projection
        init_nodes_h = np.hstack((init_nodes, np.ones((len(init_nodes), 1))))
        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, init_nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

        # temp
        us = np.where(us >= 1280, 1279, us)
        vs = np.where(vs >= 720, 719, vs)

        uvs = np.vstack((vs, us)).T
        uvs_t = tuple(map(tuple, uvs.T))

        # invert bmask for distance transform
        bmask_transformed = ndimage.distance_transform_edt(255 - bmask)
        # bmask_transformed = bmask_transformed / np.amax(bmask_transformed)
        vis = bmask_transformed[uvs_t]
        # occluded_nodes = np.where(vis > mask_dis_threshold)[0]

        # beta = 3000
        # # beta = 200
        # alpha = 1
        # gamma = 1
        # mu = 0.05
        # max_iter = 30
        # tol = 0.00001

        # include_lle = True
        # use_decoupling = True
        # use_prev_sigma2 = True
        # use_ecpd = True
        # omega = 0.00000000001

        cur_time = time.time()
        guide_nodes, nodes, sigma2 = tracking_step(filtered_pc, init_nodes, sigma2, geodesic_coord, total_len)

        init_nodes = nodes.copy()

        # add color
        nodes_rgba = struct.unpack('I', struct.pack('BBBB', 0, 0, 0, 255))[0]
        nodes_rgba_arr = np.full((len(nodes), 1), nodes_rgba)
        nodes_colored = np.hstack((nodes, nodes_rgba_arr)).astype('O')
        nodes_colored[:, 3] = nodes_colored[:, 3].astype(int)
        header.stamp = rospy.Time.now()
        converted_nodes = pcl2.create_cloud(header, fields, nodes_colored)
        nodes_pub.publish(converted_nodes)

        # add color for guide nodes
        guide_nodes_rgba = struct.unpack('I', struct.pack('BBBB', 255, 255, 255, 255))[0]
        guide_nodes_rgba_arr = np.full((len(guide_nodes), 1), guide_nodes_rgba)
        guide_nodes_colored = np.hstack((guide_nodes, guide_nodes_rgba_arr)).astype('O')
        guide_nodes_colored[:, 3] = guide_nodes_colored[:, 3].astype(int)
        header.stamp = rospy.Time.now()
        converted_guide_nodes = pcl2.create_cloud(header, fields, guide_nodes_colored)
        guide_nodes_pub.publish(converted_guide_nodes)

        # project and pub image
        nodes_h = np.hstack((nodes, np.ones((len(nodes), 1))))
        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

        tracking_img = cur_image.copy()
        for i in range (len(image_coords)):
            # draw circle
            uv = (us[i], vs[i])
            if vis[i] < mask_dis_threshold:
                cv2.circle(tracking_img, uv, 5, (0, 255, 0), -1)
            else:
                cv2.circle(tracking_img, uv, 5, (255, 0, 0), -1)

            # draw line
            if i != len(image_coords)-1:
                if vis[i] < mask_dis_threshold:
                    cv2.line(tracking_img, uv, (us[i+1], vs[i+1]), (0, 255, 0), 2)
                else:
                    cv2.line(tracking_img, uv, (us[i+1], vs[i+1]), (255, 0, 0), 2)
        
        tracking_img_msg = ros_numpy.msgify(Image, tracking_img, 'rgb8')
        tracking_img_pub.publish(tracking_img_msg)

        print(time.time() - cur_time)
        cur_time = time.time()


if __name__=='__main__':
    rospy.init_node('test', anonymous=True)

    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
    pc_sub = message_filters.Subscriber('/camera/depth/color/points', PointCloud2)

    # header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'camera_color_optical_frame'
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgba', 12, PointField.UINT32, 1)]
    pc_pub = rospy.Publisher ('/pts', PointCloud2, queue_size=10)
    init_nodes_pub = rospy.Publisher ('/init_nodes', PointCloud2, queue_size=10)
    nodes_pub = rospy.Publisher ('/nodes', PointCloud2, queue_size=10)
    guide_nodes_pub = rospy.Publisher ('/guide_nodes', PointCloud2, queue_size=10)
    tracking_img_pub = rospy.Publisher ('/tracking_img', Image, queue_size=10)
    mask_img_pub = rospy.Publisher('/mask', Image, queue_size=10)

    ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub, pc_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()