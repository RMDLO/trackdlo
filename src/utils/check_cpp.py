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
from scipy import interpolate

def pt2pt_dis_sq(pt1, pt2):
    return np.sum(np.square(pt1 - pt2))

def pt2pt_dis(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))

# assuming Y is sorted
# k -- going left for k indices, going right for k indices. a total of 2k neighbors.
def get_nearest_indices (k, Y, idx):
    if idx - k < 0:
        # use more neighbors from the other side?
        # indices_arr = np.append(np.arange(0, idx, 1), np.arange(idx+1, idx+k+1+np.abs(idx-k)))
        indices_arr = np.append(np.arange(0, idx, 1), np.arange(idx+1, idx+k+1))
        return indices_arr
    elif idx + k >= len(Y):
        last_index = len(Y) - 1
        # use more neighbots from the other side?
        # indices_arr = np.append(np.arange(idx-k-(idx+k-last_index), idx, 1), np.arange(idx+1, last_index+1, 1))
        indices_arr = np.append(np.arange(idx-k, idx, 1), np.arange(idx+1, last_index+1, 1))
        return indices_arr
    else:
        indices_arr = np.append(np.arange(idx-k, idx, 1), np.arange(idx+1, idx+k+1, 1))
        return indices_arr

def calc_LLE_weights (k, X):
    W = np.zeros((len(X), len(X)))
    for i in range (0, len(X)):
        indices = get_nearest_indices(int(k/2), X, i)
        xi, Xi = X[i], X[indices, :]
        # print("--- xi ---")
        # print(xi)
        # print("--- Xi ---")
        # print(Xi)

        component = np.full((len(Xi), len(xi)), xi).T - Xi.T
        Gi = np.matmul(component.T, component)
        # print("--- Gi ---")
        # print(Gi)

        # Gi might be singular when k is large
        if np.linalg.det(Gi) != 0:
            Gi_inv = np.linalg.inv(Gi)
        else:
            # print("Gi singular at entry " + str(i))
            epsilon = 0.00000001
            Gi_inv = np.linalg.inv(Gi + epsilon*np.identity(len(Gi)))
        # print("--- Gi_inv ---")
        # print(Gi_inv)

        wi = np.matmul(Gi_inv, np.ones((len(Xi), 1))) / np.matmul(np.matmul(np.ones(len(Xi),), Gi_inv), np.ones((len(Xi), 1)))
        # print("--- wi ---")
        # print(wi)
        
        # print("--- wi.T ---")
        # print(wi.T)
        W[i, indices] = np.squeeze(wi.T)

    return W

def ecpd_lle (X_orig,                      # input point cloud
              Y_0,                         # input nodes
              beta,                        # MCT kernel strength
              alpha,                       # MCT overall strength
              gamma,                       # LLE strength
              mu,                          # noise
              max_iter = 30,               # how many iterations EM will run
              tol = 0.00001,               # when to terminate the optimization process
              include_lle = True, 
              use_geodesic = False, 
              use_prev_sigma2 = False, 
              sigma2_0 = None,              # initial variance
              use_ecpd = False, 
              correspondence_priors = None,
              omega = None,                 # ecpd strength. DO NOT go lower than 1e-6
              kernel = 'Gaussian',          # Gaussian, Laplacian, 1st order, 2nd order
              occluded_nodes = None):       # nodes that are not in this array are either head nodes or tail nodes

    X = X_orig.copy()

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

    return G
    
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
    L = calc_LLE_weights(6, Y_0)
    H = np.matmul((np.identity(M) - L).T, np.identity(M) - L)

    # TEMP TEST
    if (occluded_nodes is not None) and (len(occluded_nodes) != 0):
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

        # determine the indices where head, tail, floating region starts/ends
        M_head = occluded_nodes[0]
        M_tail = M - 1 - occluded_nodes[-1]

        # critical nodes: M_head and M-M_tail-1
        X = np.delete(X, (max_p_nodes == M_head)|(max_p_nodes == M-M_tail-1), 0)

    if correspondence_priors is not None and len(correspondence_priors) != 0:
        additional_pc = correspondence_priors[:, 1:4]
        X = np.vstack((additional_pc, X))

    # add color
    pc_rgba = struct.unpack('I', struct.pack('BBBB', 255, 40, 40, 255))[0]
    pc_rgba_arr = np.full((len(X), 1), pc_rgba)
    filtered_pc_colored = np.hstack((X, pc_rgba_arr)).astype('O')
    filtered_pc_colored[:, 3] = filtered_pc_colored[:, 3].astype(int)

    N = len(X)
    
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
            geodesic_dists = np.zeros((M, N)).T

            for idx in max_node_smaller_index:
                geodesic_dists[idx, 0:max_p_nodes[idx]+1] = converted_node_dis[max_p_nodes[idx], 0:max_p_nodes[idx]+1] + dis_to_max_p_nodes[idx]
                geodesic_dists[idx, next_max_p_nodes[idx]:M] = converted_node_dis[next_max_p_nodes[idx], next_max_p_nodes[idx]:M] + dis_to_2nd_largest_p_nodes[idx]

            for idx in max_node_larger_index:
                geodesic_dists[idx, 0:next_max_p_nodes[idx]+1] = converted_node_dis[next_max_p_nodes[idx], 0:next_max_p_nodes[idx]+1] + dis_to_2nd_largest_p_nodes[idx]
                geodesic_dists[idx, max_p_nodes[idx]:M] = converted_node_dis[max_p_nodes[idx], max_p_nodes[idx]:M] + dis_to_max_p_nodes[idx]

            geodesic_dists = geodesic_dists.T

            P = np.exp(-np.square(geodesic_dists) / (2 * sigma2))

            if (occluded_nodes is not None) and (len(occluded_nodes) != 0):

                # modified probability distribution
                P_vis = np.zeros((M, N))

                # determine the indices where head, tail, floating region starts/ends
                M_head = occluded_nodes[0]
                M_tail = M - 1 - occluded_nodes[-1]

                P_vis_fill_head = np.zeros((M, 1))
                P_vis_fill_tail = np.zeros((M, 1))
                P_vis_fill_floating = np.zeros((M, 1))

                P_vis_fill_head[0 : M_head, 0] = 1 / M_head
                P_vis_fill_tail[M-M_tail : M, 0] = 1 / M_tail
                P_vis_fill_floating[M_head : M-M_tail, 0] = 1 / (M - M_head - M_tail)

                # fill in P_vis
                P_vis[:, (max_p_nodes >= 0)&(max_p_nodes < M_head)] = P_vis_fill_head
                P_vis[:, (max_p_nodes >= M-M_tail)&(max_p_nodes < M)] = P_vis_fill_tail
                P_vis[:, (max_p_nodes >= M_head)&(max_p_nodes < M-M_tail)] = P_vis_fill_floating
                # P_vis[:, (max_p_nodes >= 0)&(max_p_nodes <= M_head)] = P_vis_fill_head
                # P_vis[:, (max_p_nodes >= M-M_tail-1)&(max_p_nodes < M)] = P_vis_fill_tail
                # P_vis[:, (max_p_nodes > M_head)&(max_p_nodes < M-M_tail-1)] = P_vis_fill_floating

                # modify P
                P = P_vis * P

                den = np.sum(P, axis=0)
                den = np.tile(den, (M, 1))
                den[den == 0] = np.finfo(float).eps
                c = (2 * np.pi * sigma2) ** (D / 2) * mu / (1 - mu) / N
                den += c
                P = np.divide(P, den)

            else:
                den = np.sum(P, axis=0)
                den = np.tile(den, (M, 1))
                den[den == 0] = np.finfo(float).eps
                c = (2 * np.pi * sigma2) ** (D / 2)
                c = c * mu / (1 - mu)
                c = c * M / N
                den += c
                P = np.divide(P, den)

            # # original method
            # den = np.sum(P, axis=0)
            # den = np.tile(den, (M, 1))
            # den[den == 0] = np.finfo(float).eps
            # c = (2 * np.pi * sigma2) ** (D / 2)
            # c = c * mu / (1 - mu)
            # c = c * M / N
            # den += c
            # P = np.divide(P, den)

        # if occluded_nodes is not None:
        #     print(occluded_nodes)
        #     P[occluded_nodes] = 0

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
            rospy.loginfo('Iteration until covnergence: ' + str(i) + '. Kernel used: ' + kernel)
            break
        else:
            Y = Y_0 + np.matmul(G, W)

        if i == max_iter - 1:
            # print error messages if optimization did not compile
            rospy.logerr('Optimization did not converge! ' + 'Kernel used: ' + kernel)

    return Y, sigma2


if __name__=='__main__':
    m1 = np.arange(0, 5*3, 1).reshape((5, 3)) / 100
    # print(m1)

    # # ----- test LLE weights -----
    # cur_time = time.time()
    # out = calc_LLE_weights(2, m1)
    # print(time.time() - cur_time)
    # # print(out)

    # test ecpd
    print(ecpd_lle(np.array([0, 0, 0]), m1, 0.3, None, None, None, kernel="2nd order"))