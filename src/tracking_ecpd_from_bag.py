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

import time
import pickle as pkl

import message_filters
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from scipy import ndimage

def pt2pt_dis_sq(pt1, pt2):
    return np.sum(np.square(pt1 - pt2))

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

def cpd_lle (X, 
             Y_0, 
             beta, 
             alpha, 
             gamma, 
             mu, 
             max_iter, 
             tol, 
             include_lle=True, 
             use_decoupling=False, 
             use_prev_sigma2=False, 
             sigma2_0=None, 
             use_ecpd=False, 
             omega=None, 
             threshold=None,
             consider_pvis=False,
             occluded_nodes=None):

    # define params
    M = len(Y_0)
    N = len(X)
    # D = len(X[0])
    D = 3

    # initialization
    # faster G calculation
    diff = Y_0[:, None, :] - Y_0[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)

    converted_node_dis = []
    if not use_decoupling:
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

        # # TEST
        # if threshold is not None:
        #     dis_transform_coeff = 1
        #     converted_node_dis_sq = -np.exp(-(dis_transform_coeff * converted_node_dis_sq - np.log(threshold))) + threshold

        G = np.exp(-converted_node_dis_sq / (2 * beta**2))
        # G = np.exp(-np.square(converted_node_dis) / np.sqrt(converted_node_dis) / (2 * beta**2))

        # temp = np.power(converted_node_dis + np.identity(M), 1.2) - np.identity(M)
        # temp = converted_node_dis
        # G = np.exp(-temp / (2 * beta**2))
    
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
    print('time taken = ', time.time() - cur_time)
    
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

        if consider_pvis:
            P[occluded_nodes] = 0

        if use_decoupling:
            max_p_nodes = np.argmax(P, axis=0)
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

        Pt1 = np.sum(P, axis=0)
        P1 = np.sum(P, axis=1)
        Np = np.sum(P1)
        PX = np.matmul(P, X)
    
        # M step
        if include_lle:
            if use_ecpd:
                P_tilde = np.zeros((M, N))
                pt_node_correspondence = np.argmax(P, axis=0)
                
                for node_num in range (0, M):
                    node_num_pts_indices = np.where(pt_node_correspondence == node_num)
                    P_tilde[node_num, node_num_pts_indices] = 1
                
                if consider_pvis:
                    P_tilde[occluded_nodes] = 0

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
                
                if consider_pvis:
                    P_tilde[occluded_nodes] = 0

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
    
    print('sigma2 after reg = ', sigma2)
    return Y, sigma2

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

    start_idx = 15

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

cur_image = []
cur_image_arr = []
def update_rgb (data):
    global cur_image
    global cur_image_arr
    temp = ros_numpy.numpify(data)
    if len(cur_image_arr) <= 3:
        cur_image_arr.append(temp)
        cur_image = cur_image_arr[0]
    else:
        cur_image_arr.append(temp)
        cur_image = cur_image_arr[0]
        cur_image_arr.pop(0)

bmask = []
mask = []
def update_mask (data):
    global bmask
    global mask
    mask = ros_numpy.numpify(data)
    bmask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

saved = False
initialized = False
init_nodes = []
nodes = []
sigma2 = 0
cur_time = time.time()
def callback (pc):
    global cur_image
    global bmask
    global mask

    global saved
    global initialized
    global init_nodes
    global nodes
    global sigma2
    global cur_time

    proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                            [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                            [             0.0,              0.0,               1.0, 0.0]])

    # process point cloud
    pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc)
    filtered_pc = ros_numpy.point_cloud2.get_xyz_points(pc_data)
    # filtered_pc = filtered_pc.reshape((720, 1280, 3))

    # register nodes
    if not initialized:
        init_nodes, sigma2 = register(filtered_pc, 35, mu=0.05, max_iter=100)
        init_nodes = np.array(sort_pts(init_nodes.tolist()))
        initialized = True
        # header.stamp = rospy.Time.now()
        # converted_init_nodes = pcl2.create_cloud(header, fields, init_nodes)
        # init_nodes_pub.publish(converted_init_nodes)

    # cpd
    if initialized:
        # determined which nodes are occluded from mask information
        mask_dis_threshold = 20
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
        occluded_nodes = np.where(vis > mask_dis_threshold)[0]

        beta = 1
        # beta = 3500
        alpha = 1
        gamma = 1
        mu = 0.05
        max_iter = 30
        tol = 0.00001

        include_lle = True
        use_decoupling = True
        use_prev_sigma2 = True
        use_ecpd = True
        omega = 0.0000001
        threshold = None
        consider_pvis = False

        cur_time = time.time()
        nodes, sigma2 = cpd_lle(X = filtered_pc, 
                            Y_0 = init_nodes, 
                            beta = beta,       # beta   : a constant representing the strength of interaction between points
                            alpha = alpha,     # alpha  : a constant regulating the strength of smoothing
                            gamma = gamma,     # gamma  : a constant regulating the strength of LLE
                            mu = mu,           # mu     : a constant representing how noisy the input is (0 - 1)
                            max_iter = max_iter, 
                            tol = tol, 
                            include_lle = include_lle, 
                            use_decoupling = use_decoupling, 
                            use_prev_sigma2 = use_prev_sigma2, 
                            sigma2_0 = sigma2,
                            use_ecpd = use_ecpd,
                            omega = omega,
                            threshold = threshold,
                            consider_pvis = consider_pvis,
                            occluded_nodes = occluded_nodes)

        init_nodes = nodes.copy()

        # add color
        nodes_rgba = struct.unpack('I', struct.pack('BBBB', 0, 0, 0, 255))[0]
        nodes_rgba_arr = np.full((len(nodes), 1), nodes_rgba)
        nodes_colored = np.hstack((nodes, nodes_rgba_arr)).astype('O')
        nodes_colored[:, 3] = nodes_colored[:, 3].astype(int)
        header.stamp = rospy.Time.now()
        converted_nodes = pcl2.create_cloud(header, fields, nodes_colored)
        nodes_pub.publish(converted_nodes)

        # project and pub image
        nodes_h = np.hstack((nodes, np.ones((len(nodes), 1))))
        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

        tracking_img = cur_image.copy()
        # tracking_img = mask.copy()
        for i in range (len(image_coords)):
            try:
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
            except:
                print('image coordinate out of range!')
        
        tracking_img_msg = ros_numpy.msgify(Image, tracking_img, 'rgb8')
        tracking_img_pub.publish(tracking_img_msg)

        print(time.time() - cur_time)
        cur_time = time.time()


if __name__=='__main__':
    rospy.init_node('test', anonymous=True)

    rospy.Subscriber('/camera/color/image_raw', Image, update_rgb)
    rospy.Subscriber('/mask', Image, update_mask)
    filtered_pc_sub = message_filters.Subscriber('/pts', PointCloud2)

    # header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'camera_color_optical_frame'
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgba', 12, PointField.UINT32, 1)]
    init_nodes_pub = rospy.Publisher ('/init_nodes', PointCloud2, queue_size=10)
    nodes_pub = rospy.Publisher ('/nodes', PointCloud2, queue_size=10)
    tracking_img_pub = rospy.Publisher ('/tracking_img', Image, queue_size=10)

    ts = message_filters.TimeSynchronizer([filtered_pc_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()