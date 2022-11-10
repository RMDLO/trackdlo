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

    return new_Y, new_s

def sort_pts_mst (pts_orig):

    INF = 999999
    diff = pts_orig[:, None, :] - pts_orig[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    N = len(diff)
    G = diff.copy()
    selected_node = np.zeros(N,).tolist()

    no_edge = 0
    selected_node[0] = True
    sorted_pts = []

    init_a = None
    reverse = False
    while (no_edge < N - 1):
        
        minimum = INF
        a = 0
        b = 0
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and G[m][n]):  
                        # not in selected and there is an edge
                        if minimum > G[m][n]:
                            minimum = G[m][n]
                            a = m
                            b = n

        if len(sorted_pts) == 0:
            sorted_pts.append(pts_orig[a])
            sorted_pts.append(pts_orig[b])
            init_a = a
        else:
            if a == init_a:
                reverse = True
            if reverse:
                # switch direction
                sorted_pts.insert(0, pts_orig[b])
            else:
                sorted_pts.append(pts_orig[b])
        selected_node[b] = True
        no_edge += 1

    return np.array(sorted_pts)

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

def ecpd_lle (X_orig,                      # input point cloud
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
              occluded_nodes = None):       # nodes that are not in this array are either head nodes or tail nodes

    X = X_orig.copy()
    if correspondence_priors is not None and len(correspondence_priors) != 0:
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

        # if (occluded_nodes is not None) and (len(occluded_nodes) != 0):
        #     # determine the indices where head, tail, floating region starts/ends
        #     M_head = occluded_nodes[0]
        #     M_tail = M - 1 - occluded_nodes[-1]

        #     # critical nodes: M_head and M-M_tail-1
        #     X = np.delete(X, (max_p_nodes == M_head)|(max_p_nodes == M-M_tail-1), 0)
        #     # P = np.delete(P, (max_p_nodes == M_head)|(max_p_nodes == M-M_tail-1), 1)
        #     print('deleted', N - len(X))
        #     N = len(X)

        # pts_dis_sq = np.sum((X[None, :, :] - Y[:, None, :]) ** 2, axis=2)
        # c = (2 * np.pi * sigma2) ** (D / 2)
        # c = c * mu / (1 - mu)
        # c = c * M / N
        # P = np.exp(-pts_dis_sq / (2 * sigma2))
        # den = np.sum(P, axis=0)
        # den = np.tile(den, (M, 1))
        # den[den == 0] = np.finfo(float).eps
        # den += c
        # P = np.divide(P, den)
        # max_p_nodes = np.argmax(P, axis=0)

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
                print('--- modified p ---')
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

                # # critical nodes: M_head and M-M_tail-1
                # X = np.delete(X, (max_p_nodes == M_head)|(max_p_nodes == M-M_tail-1), 0)
                # P_vis = np.delete(P_vis, (max_p_nodes == M_head)|(max_p_nodes == M-M_tail-1), 1)
                # P = np.delete(P, (max_p_nodes == M_head)|(max_p_nodes == M-M_tail-1), 1)
                # print('deleted', N - len(X))
                # N = len(X)

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

        # add color
        pc_rgba = struct.unpack('I', struct.pack('BBBB', 255, 40, 40, 255))[0]
        pc_rgba_arr = np.full((len(X), 1), pc_rgba)
        filtered_pc_colored = np.hstack((X, pc_rgba_arr)).astype('O')
        filtered_pc_colored[:, 3] = filtered_pc_colored[:, 3].astype(int)

        # filtered_pc = filtered_pc.reshape((len(filtered_pc)*len(filtered_pc[0]), 3))
        header.stamp = rospy.Time.now()
        converted_points = pcl2.create_cloud(header, fields, filtered_pc_colored)
        pc_pub.publish(converted_points)

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

        if i == max_iter - 1:
            # print error messages if optimization did not compile
            rospy.logerr('Optimization did not converge!')

    return Y, sigma2

def pre_process (X, Y_0, geodesic_coord, total_len, bmask, sigma2_0):

    proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                            [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                            [             0.0,              0.0,               1.0, 0.0]])

    guide_nodes, _ = ecpd_lle(X, Y_0, 10, 1, 1, 0.05, 30, 0.00001, True, True, use_prev_sigma2=False, sigma2_0=None, kernel = 'Laplacian')

    # determine which head node is occluded, if any
    head_visible = False
    tail_visible = False

    if pt2pt_dis(guide_nodes[0], Y_0[0]) < 0.01:
        head_visible = True
    if pt2pt_dis(guide_nodes[-1], Y_0[-1]) < 0.01:
        tail_visible = True

    if not head_visible and not tail_visible:
        if pt2pt_dis(guide_nodes[0], Y_0[0]) < pt2pt_dis(guide_nodes[-1], Y_0[-1]):
            head_visible = True
        else:
            tail_visible = True

    cur_total_len = np.sum(np.sqrt(np.sum(np.square(np.diff(guide_nodes, axis=0)), axis=1)))

    print('tail displacement = ', pt2pt_dis(guide_nodes[-1], Y_0[-1]))
    print('head displacement = ', pt2pt_dis(guide_nodes[0], Y_0[0]))
    print('length difference = ', abs(cur_total_len - total_len))

    # visible_dist = np.sum(np.sqrt(np.sum(np.square(np.diff(guide_nodes, axis=0)), axis=1)))
    correspondence_priors = None
    occluded_nodes = None

    mask_dis_threshold = 10

    if abs(cur_total_len - total_len) < 0.01: # (head_visible and tail_visible) or 
        print('head visible and tail visible or the same len')
        correspondence_priors = []
        correspondence_priors.append(np.append(np.array([0]), guide_nodes[0]))
        correspondence_priors.append(np.append(np.array([len(guide_nodes)-1]), guide_nodes[-1]))
    
    # elif head_visible and tail_visible:
    elif head_visible and tail_visible: # but length condiiton not met - middle part is occluded
        print('head and tail visible but total length changed')

        # first need to determine which portion of the guide nodes are actual useful data (not occupying empty space)
        # determined which nodes are occluded from mask information
        # projection
        guide_nodes_h = np.hstack((guide_nodes, np.ones((len(guide_nodes), 1))))
        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, guide_nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)
        # cap uv to be within (1280, 720)
        us = np.where(us >= 1280, 1279, us)
        vs = np.where(vs >= 720, 719, vs)
        uvs = np.vstack((vs, us)).T
        uvs_t = tuple(map(tuple, uvs.T))
        # invert bmask for distance transform
        bmask_transformed = ndimage.distance_transform_edt(255 - bmask)
        # bmask_transformed = bmask_transformed / np.amax(bmask_transformed)
        vis = bmask_transformed[uvs_t]
        valid_guide_nodes_indices = np.where(vis < mask_dis_threshold)[0]

        # TEMP
        cur_time = time.time()

        # determine a set of nodes for head and a set of nodes for tail
        valid_head_node_indices = []
        for node_idx in range (0, len(guide_nodes)):
            if node_idx in valid_guide_nodes_indices:
                # valid_head_nodes.append(guide_nodes[node_idx])
                valid_head_node_indices.append(node_idx)
            else: 
                break
        if len(valid_head_node_indices) != 0:
            valid_head_nodes = guide_nodes[np.array(valid_head_node_indices)]
        else:
            valid_head_nodes = []
            print('error! no valid head nodes')

        valid_tail_node_indices = []
        for node_idx in range (len(guide_nodes)-1, -1, -1):
            if node_idx in valid_guide_nodes_indices:
                # valid_tail_nodes.append(guide_nodes[node_idx])
                valid_tail_node_indices.append(node_idx)
            else:
                break

        if len(valid_tail_node_indices) != 0:
            valid_tail_nodes = guide_nodes[np.array(valid_tail_node_indices)] # valid tail node is reversed, starting from the end
        else:
            valid_tail_nodes = []
            print('error! no valid tail nodes')

        # initialize a variable for last visible head index and last visible tail index
        last_visible_index_head = None
        last_visible_index_tail = None

        # ----- head visible part -----
        correspondence_priors_head = []

        # if only one head node is visible, should not fit spline
        if len(valid_head_nodes) <= 2:
            correspondence_priors_head = np.hstack((valid_head_node_indices[0], valid_head_nodes[0]))
            last_visible_index_head = 0

        else:
            num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(valid_head_nodes, axis=0)), axis=1)))/0.001)
            tck, u = interpolate.splprep(valid_head_nodes.T, s=0.0001)
            u_fine = np.linspace(0,1,num_true_pts)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T
            total_spline_len = np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1)))

            last_visible_index_head = len(geodesic_coord[geodesic_coord < total_spline_len]) - 1

            # geodesic coord is 1D
            correspondence_priors_head = np.vstack((np.arange(0, last_visible_index_head+1), spline_pts[(geodesic_coord[0:last_visible_index_head+1]*1000).astype(int)].T)).T
            # occluded_nodes = np.arange(last_visible_index_head+1, len(Y_0), 1)
        
        # ----- tail visible part -----
        correspondence_priors_tail = []

        # if not enough tail node is visible, should not fit spline
        if len(valid_tail_nodes) <= 2:
            correspondence_priors_tail = np.hstack((valid_tail_node_indices[0], valid_tail_nodes[0]))
            last_visible_index_tail = len(Y_0) - 1

        else:
            num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(valid_tail_nodes, axis=0)), axis=1)))/0.001)
            tck, u = interpolate.splprep(valid_tail_nodes.T, s=0.0001)
            u_fine = np.linspace(0,1,num_true_pts)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T
            total_spline_len = np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1)))

            geodesic_coord_from_tail = np.abs(geodesic_coord - geodesic_coord[-1]).tolist()
            geodesic_coord_from_tail.reverse()
            geodesic_coord_from_tail = np.array(geodesic_coord_from_tail)

            last_visible_index_tail = len(Y_0) - len(geodesic_coord_from_tail[geodesic_coord_from_tail < total_spline_len])

            # geodesic coord is 1D
            correspondence_priors_tail = np.vstack((np.arange(len(Y_0)-1, last_visible_index_tail-1, -1), spline_pts[(geodesic_coord_from_tail[0:len(geodesic_coord_from_tail[geodesic_coord_from_tail < total_spline_len])]*1000).astype(int)].T)).T        
            # occluded_nodes = np.arange(0, last_visible_index_tail, 1)

        # compile occluded nodes
        occluded_nodes = np.arange(last_visible_index_head+1, last_visible_index_tail, 1)
        correspondence_priors = np.vstack((correspondence_priors_head, correspondence_priors_tail))

        # TEMP
        print('pre-process time taken:', time.time()-cur_time)

    elif head_visible and not tail_visible:

        # first need to determine which portion of the guide nodes are actual useful data (not occupying empty space)
        # determined which nodes are occluded from mask information
        # projection
        guide_nodes_h = np.hstack((guide_nodes, np.ones((len(guide_nodes), 1))))
        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, guide_nodes_h.T).T
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
        valid_guide_nodes_indices = np.where(vis < mask_dis_threshold)[0]

        valid_head_node_indices = []
        for node_idx in range (0, len(guide_nodes)):
            if node_idx in valid_guide_nodes_indices:
                # valid_head_nodes.append(guide_nodes[node_idx])
                valid_head_node_indices.append(node_idx)
            else: 
                break
        if len(valid_head_node_indices) != 0:
            valid_head_nodes = guide_nodes[np.array(valid_head_node_indices)]
        else:
            valid_head_nodes = []
            print('error! no valid head nodes')

        print('head visible')
        correspondence_priors = []

        # if only one head node is visible, should not fit spline
        if len(valid_head_nodes) == 1:
            correspondence_priors_head = np.hstack((valid_head_node_indices[0], valid_head_nodes[0]))
            last_visible_index_head = 0

        else:
            num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(valid_head_nodes, axis=0)), axis=1)))/0.001)
            tck, u = interpolate.splprep(valid_head_nodes.T, s=0.0001)
            u_fine = np.linspace(0,1,num_true_pts)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T
            total_spline_len = np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1)))

            last_visible_index_head = len(geodesic_coord[geodesic_coord < total_spline_len]) - 1

            # geodesic coord is 1D
            correspondence_priors = np.vstack((np.arange(0, last_visible_index_head+1), spline_pts[(geodesic_coord[0:last_visible_index_head+1]*1000).astype(int)].T)).T
        
        occluded_nodes = np.arange(last_visible_index_head+1, len(Y_0), 1)

    elif tail_visible and not head_visible:

        # first need to determine which portion of the guide nodes are actual useful data (not occupying empty space)
        # determined which nodes are occluded from mask information
        # projection
        guide_nodes_h = np.hstack((guide_nodes, np.ones((len(guide_nodes), 1))))
        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, guide_nodes_h.T).T
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
        valid_guide_nodes_indices = np.where(vis < mask_dis_threshold)[0]

        valid_tail_node_indices = []
        for node_idx in range (len(guide_nodes)-1, -1, -1):
            if node_idx in valid_guide_nodes_indices:
                # valid_tail_nodes.append(guide_nodes[node_idx])
                valid_tail_node_indices.append(node_idx)
            else:
                break

        if len(valid_tail_node_indices) != 0:
            valid_tail_nodes = guide_nodes[np.array(valid_tail_node_indices)] # valid tail node is reversed, starting from the end
        else:
            valid_tail_nodes = []
            print('error! no valid tail nodes')

        print('tail visible')
        correspondence_priors = []

        # if only one tail node is visible, should not fit spline
        if len(valid_tail_nodes) == 1:
            correspondence_priors_tail = np.hstack((valid_tail_node_indices[0], valid_tail_nodes[0]))
            last_visible_index_tail = len(Y_0) - 1

        else:
            num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(valid_tail_nodes, axis=0)), axis=1)))/0.001)
            tck, u = interpolate.splprep(valid_tail_nodes.T, s=0.0001)
            u_fine = np.linspace(0,1,num_true_pts)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T
            total_spline_len = np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1)))

            geodesic_coord_from_tail = np.abs(geodesic_coord - geodesic_coord[-1]).tolist()
            geodesic_coord_from_tail.reverse()
            geodesic_coord_from_tail = np.array(geodesic_coord_from_tail)

            last_visible_index_tail = len(Y_0) - len(geodesic_coord_from_tail[geodesic_coord_from_tail < total_spline_len])

            # geodesic coord is 1D
            correspondence_priors = np.vstack((np.arange(len(Y_0)-1, last_visible_index_tail-1, -1), spline_pts[(geodesic_coord_from_tail[0:len(geodesic_coord_from_tail[geodesic_coord_from_tail < total_spline_len])]*1000).astype(int)].T)).T        
            
        occluded_nodes = np.arange(0, last_visible_index_tail, 1)
    
    # if none of the above condition is satisfied
    else:
        print('error!')

    return guide_nodes, np.array(correspondence_priors), occluded_nodes

def tracking_step (X, Y_0, sigma2_0, geodesic_coord, total_len, bmask):
    guide_nodes, correspondence_priors, occluded_nodes = pre_process(X, Y_0, geodesic_coord, total_len, bmask, sigma2_0)
    Y, sigma2 = ecpd_lle(X, Y_0, 7, 1, 1, 0.0, 30, 0.00001, True, True, True, sigma2_0, True, correspondence_priors, omega=0.000001, kernel='1st order', occluded_nodes=occluded_nodes)
    # Y, sigma2 = ecpd_lle(X, Y_0, 1, 1, 1, 0.1, 30, 0.00001, True, True, True, sigma2_0, True, correspondence_priors, 0.01, 'Gaussian', occluded_nodes)
    # Y, sigma2 = ecpd_lle(X, Y_0, 2, 1, 1, 0.1, 30, 0.00001, True, True, True, sigma2_0, True, correspondence_priors, 0.001, '2nd order', occluded_nodes)

    return correspondence_priors[:, 1:4], Y, sigma2  # correspondence_priors[:, 1:4]

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

    # publish mask
    mask_img_msg = ros_numpy.msgify(Image, mask, 'rgb8')
    mask_img_pub.publish(mask_img_msg)

    mask = (mask/255).astype(int)

    filtered_pc = cur_pc*mask
    filtered_pc = filtered_pc[((filtered_pc[:, :, 0] != 0) | (filtered_pc[:, :, 1] != 0) | (filtered_pc[:, :, 2] != 0))]
    # filtered_pc = filtered_pc[filtered_pc[:, 2] < 0.705]
    filtered_pc = filtered_pc[filtered_pc[:, 2] > 0.4]

    # downsample with open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_pc)
    downpcd = pcd.voxel_down_sample(voxel_size=0.005)
    filtered_pc = np.asarray(downpcd.points)

    # # add color
    # pc_rgba = struct.unpack('I', struct.pack('BBBB', 255, 40, 40, 255))[0]
    # pc_rgba_arr = np.full((len(filtered_pc), 1), pc_rgba)
    # filtered_pc_colored = np.hstack((filtered_pc, pc_rgba_arr)).astype('O')
    # filtered_pc_colored[:, 3] = filtered_pc_colored[:, 3].astype(int)

    # # filtered_pc = filtered_pc.reshape((len(filtered_pc)*len(filtered_pc[0]), 3))
    # header.stamp = rospy.Time.now()
    # converted_points = pcl2.create_cloud(header, fields, filtered_pc_colored)
    # pc_pub.publish(converted_points)

    # register nodes
    if not initialized:
        init_nodes, sigma2 = register(filtered_pc, 25, mu=0.05, max_iter=100)
        init_nodes = sort_pts_mst(init_nodes)

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
        mask_dis_threshold = 10
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

        cur_time = time.time()
        guide_nodes, nodes, sigma2 = tracking_step(filtered_pc, init_nodes, sigma2, geodesic_coord, total_len, bmask)

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
        # nodes_h = np.hstack((guide_nodes, np.ones((len(nodes), 1)))) # TEMP

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