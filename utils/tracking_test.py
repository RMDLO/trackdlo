#!/usr/bin/env python3

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

import message_filters
import open3d as o3d
from scipy import ndimage

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from scipy.spatial.transform import Rotation as R

proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                        [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                        [             0.0,              0.0,               1.0, 0.0]])

def pt2pt_dis_sq(pt1, pt2):
    return np.sum(np.square(pt1 - pt2))

def pt2pt_dis(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))

occlusion_mask_rgb = None
def update_occlusion_mask(data):
	global occlusion_mask_rgb
	occlusion_mask_rgb = ros_numpy.numpify(data)

# original post: https://stackoverflow.com/a/59204638
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def ndarray2MarkerArray (Y, marker_frame, node_color, line_color):
    results = MarkerArray()
    for i in range (0, len(Y)):
        cur_node_result = Marker()
        cur_node_result.header.frame_id = marker_frame
        cur_node_result.type = Marker.SPHERE
        cur_node_result.action = Marker.ADD
        cur_node_result.ns = "node_results" + str(i)
        cur_node_result.id = i

        cur_node_result.pose.position.x = Y[i, 0]
        cur_node_result.pose.position.y = Y[i, 1]
        cur_node_result.pose.position.z = Y[i, 2]
        cur_node_result.pose.orientation.w = 1.0
        cur_node_result.pose.orientation.x = 0.0
        cur_node_result.pose.orientation.y = 0.0
        cur_node_result.pose.orientation.z = 0.0

        cur_node_result.scale.x = 0.01
        cur_node_result.scale.y = 0.01
        cur_node_result.scale.z = 0.01
        cur_node_result.color.r = node_color[0]
        cur_node_result.color.g = node_color[1]
        cur_node_result.color.b = node_color[2]
        cur_node_result.color.a = node_color[3]

        results.markers.append(cur_node_result)

        if i == len(Y)-1:
            break

        cur_line_result = Marker()
        cur_line_result.header.frame_id = marker_frame
        cur_line_result.type = Marker.CYLINDER
        cur_line_result.action = Marker.ADD
        cur_line_result.ns = "line_results" + str(i)
        cur_line_result.id = i

        cur_line_result.pose.position.x = ((Y[i] + Y[i+1])/2)[0]
        cur_line_result.pose.position.y = ((Y[i] + Y[i+1])/2)[1]
        cur_line_result.pose.position.z = ((Y[i] + Y[i+1])/2)[2]

        rot_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), (Y[i+1]-Y[i])/pt2pt_dis(Y[i+1], Y[i])) 
        r = R.from_matrix(rot_matrix)
        x = r.as_quat()[0]
        y = r.as_quat()[1]
        z = r.as_quat()[2]
        w = r.as_quat()[3]

        cur_line_result.pose.orientation.w = w
        cur_line_result.pose.orientation.x = x
        cur_line_result.pose.orientation.y = y
        cur_line_result.pose.orientation.z = z
        cur_line_result.scale.x = 0.005
        cur_line_result.scale.y = 0.005
        cur_line_result.scale.z = pt2pt_dis(Y[i], Y[i+1])
        cur_line_result.color.r = line_color[0]
        cur_line_result.color.g = line_color[1]
        cur_line_result.color.b = line_color[2]
        cur_line_result.color.a = line_color[3]

        results.markers.append(cur_line_result)
    
    return results

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
    
    for it in range (max_iter):
        prev_Y, prev_s = new_Y, new_s
        new_Y, new_s = get_estimates(prev_Y, prev_s)

    return new_Y, new_s

def sort_pts (Y_0):
    diff = Y_0[:, None, :] - Y_0[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)

    N = len(diff)
    G = diff.copy()

    selected_node = np.zeros(N,).tolist()
    selected_node[0] = True
    Y_0_sorted = []
        
    reverse = 0
    counter = 0
    reverse_on = 0
    insertion_counter = 0
    last_visited_b = 0
    while (counter < N - 1):
        
        minimum = 999999
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

        if len(Y_0_sorted) == 0:
            Y_0_sorted.append(Y_0[a].tolist())
            Y_0_sorted.append(Y_0[b].tolist())
        else:
            if last_visited_b != a:
                reverse += 1
                reverse_on = a
                insertion_counter = 0

            if reverse % 2 == 1:
                # switch direction
                Y_0_sorted.insert(Y_0_sorted.index(Y_0[a].tolist()), Y_0[b].tolist())
            elif reverse != 0:
                Y_0_sorted.insert(Y_0_sorted.index(Y_0[reverse_on].tolist())+1+insertion_counter, Y_0[b].tolist())
                insertion_counter += 1
            else:
                Y_0_sorted.append(Y_0[b].tolist())

        last_visited_b = b
        selected_node[b] = True

        counter += 1

    return np.array(Y_0_sorted)

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

def cpd_lle (X, Y_0, beta, alpha, gamma, mu, max_iter=50, tol=0.00001, include_lle=True, use_geodesic=False, use_prev_sigma2=False, sigma2_0=None):

    # define params
    M = len(Y_0)
    N = len(X)
    D = len(X[0])

    # initialization
    # faster G calculation
    diff = Y_0[:, None, :] - Y_0[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)

    converted_node_dis = []
    if not use_geodesic:
        # Gaussian Kernel
        G = np.exp(-diff / (2 * beta**2))
    else:
        # compute the geodesic distances between nodes
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

        # Gaussian Kernel
        G = np.exp(-converted_node_dis_sq / (2 * beta**2))

        # temp
        # G[converted_node_dis > 0.07] = 0
    
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
    
    # loop until convergence or max_iter reached
    for it in range (0, max_iter):

        # ----- E step: compute posteriori probability matrix P -----
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

        # if use geodesic, overwrite P
        # this section looks long, but it is simply replacing the Euclidean distances in P with geodesic distances
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

        Pt1 = np.sum(P, axis=0)
        P1 = np.sum(P, axis=1)
        Np = np.sum(P1)
        PX = np.matmul(P, X)

        # print(Pt1)
    
        # ----- M step: solve for new weights and variance -----
        if include_lle:
            A_matrix = np.matmul(np.diag(P1), G) + alpha * sigma2 * np.identity(M) + sigma2 * gamma * np.matmul(H, G)
            B_matrix = PX - np.matmul(np.diag(P1) + sigma2*gamma*H, Y_0)
        else:
            A_matrix = np.matmul(np.diag(P1), G) + alpha * sigma2 * np.identity(M)
            B_matrix = PX - np.matmul(np.diag(P1), Y_0)

        # solve for W
        W = np.linalg.solve(A_matrix, B_matrix)

        T = Y_0 + np.matmul(G, W)
        trXtdPt1X = np.trace(np.matmul(np.matmul(X.T, np.diag(Pt1)), X))
        trPXtT = np.trace(np.matmul(PX.T, T))
        trTtdP1T = np.trace(np.matmul(np.matmul(T.T, np.diag(P1)), T))

        # solve for sigma^2
        sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D)

        # update Y
        if pt2pt_dis_sq(Y, Y_0 + np.matmul(G, W)) < tol:
            # if converged, break loop
            Y = Y_0 + np.matmul(G, W)
            print("iteration until convergence:", it)
            break
        else:
            # keep going until max iteration is reached
            Y = Y_0 + np.matmul(G, W)

            if it == max_iter - 1:
                print("did not converge!")
    
    return Y, sigma2

initialized = False
use_eval_rope = True
pub_tracking_img = True
init_nodes = []
nodes = []
guide_nodes_Y_0 = []
sigma2 = 0
guide_nodes_sigma2_0 = 0
total_len = 0
geodesic_coord = []
def callback (rgb, pc):
    global initialized
    global init_nodes, nodes, sigma2
    global total_len, geodesic_coord
    global guide_nodes_Y_0, guide_nodes_sigma2_0
    global params, read_params
    global occlusion_mask_rgb

    # log time
    cur_time_cb = time.time()
    print('----------')

    # process rgb image
    cur_image = ros_numpy.numpify(rgb)
    # cur_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_RGB2HSV)

    # process point cloud
    pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc)
    cur_pc = ros_numpy.point_cloud2.get_xyz_points(pc_data)
    cur_pc = cur_pc.reshape((720, 1280, 3))

    # process opencv mask
    if occlusion_mask_rgb is None:
        occlusion_mask_rgb = np.ones(cur_image.shape).astype('uint8')*255
    occlusion_mask = cv2.cvtColor(occlusion_mask_rgb.copy(), cv2.COLOR_RGB2GRAY)

    if not use_eval_rope:
        # color thresholding
        lower = (90, 90, 90)
        upper = (120, 255, 255)
        mask = cv2.inRange(hsv_image, lower, upper)
    else:
        # color thresholding
        # --- rope blue ---
        lower = (90, 60, 40)
        upper = (130, 255, 255)
        mask_dlo = cv2.inRange(hsv_image, lower, upper).astype('uint8')

        # --- tape red ---
        lower = (130, 60, 40)
        upper = (255, 255, 255)
        mask_red_1 = cv2.inRange(hsv_image, lower, upper).astype('uint8')
        lower = (0, 60, 40)
        upper = (10, 255, 255)
        mask_red_2 = cv2.inRange(hsv_image, lower, upper).astype('uint8')
        mask_marker = cv2.bitwise_or(mask_red_1.copy(), mask_red_2.copy()).astype('uint8')

        # combine masks
        mask = cv2.bitwise_or(mask_marker.copy(), mask_dlo.copy())
        mask = cv2.bitwise_and(mask.copy(), occlusion_mask.copy())

    bmask = mask.copy() # for checking visibility, max = 255
    mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    # publish mask
    mask_img_msg = ros_numpy.msgify(Image, mask, 'rgb8')
    mask_img_pub.publish(mask_img_msg)

    mask = (mask/255).astype(int)

    filtered_pc = cur_pc*mask
    filtered_pc = filtered_pc[((filtered_pc[:, :, 0] != 0) | (filtered_pc[:, :, 1] != 0) | (filtered_pc[:, :, 2] != 0))]
    # filtered_pc = filtered_pc[filtered_pc[:, 2] < 0.705]
    filtered_pc = filtered_pc[filtered_pc[:, 2] > 0.58]

    # downsample with open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_pc)
    downpcd = pcd.voxel_down_sample(voxel_size=0.005)
    filtered_pc = np.asarray(downpcd.points)

    rospy.loginfo("Downsampled point cloud size: " + str(len(filtered_pc)))

    # add color
    pc_rgba = struct.unpack('I', struct.pack('BBBB', 255, 40, 40, 255))[0]
    pc_rgba_arr = np.full((len(filtered_pc), 1), pc_rgba)
    filtered_pc_colored = np.hstack((filtered_pc, pc_rgba_arr)).astype('O')
    filtered_pc_colored[:, 3] = filtered_pc_colored[:, 3].astype(int)

    # filtered_pc = filtered_pc.reshape((len(filtered_pc)*len(filtered_pc[0]), 3))
    header.stamp = rospy.Time.now()
    converted_points = pcl2.create_cloud(header, fields, filtered_pc_colored)
    pc_pub.publish(converted_points)

    rospy.logwarn('callback before initialized: ' + str((time.time() - cur_time_cb)*1000) + ' ms')

    # register nodes
    if not initialized:

        init_nodes, sigma2 = register(filtered_pc, 40, 0.05, max_iter=100)
        init_nodes = sort_pts(init_nodes)

        nodes = init_nodes.copy()

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
        # nodes_pub.publish(converted_init_nodes)

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

        # log time
        cur_time = time.time()
        nodes, sigma2 = cpd_lle(filtered_pc, nodes, 0.7, 5, 1, 0.05, 50, 0.00001, True, False, False, sigma2)
        rospy.logwarn('tracking_step total: ' + str((time.time() - cur_time)*1000) + ' ms')

        init_nodes = nodes.copy()

        results = ndarray2MarkerArray(nodes, "camera_color_optical_frame", [255, 150, 0, 0.75], [0, 255, 0, 0.75])
        results_pub.publish(results)

        if pub_tracking_img:
            # project and pub tracking image
            nodes_h = np.hstack((nodes, np.ones((len(nodes), 1))))

            # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
            image_coords = np.matmul(proj_matrix, nodes_h.T).T
            us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
            vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

            cur_image_masked = cv2.bitwise_and(cur_image, occlusion_mask_rgb)
            tracking_img = (cur_image*0.5 + cur_image_masked*0.5).astype(np.uint8)

            for i in range (len(image_coords)):
                # draw circle
                uv = (us[i], vs[i])
                if vis[i] < mask_dis_threshold:
                    cv2.circle(tracking_img, uv, 5, (255, 150, 0), -1)
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

        rospy.logwarn('callback total: ' + str((time.time() - cur_time_cb)*1000) + ' ms')

if __name__=='__main__':

    rospy.init_node('tracking_test', anonymous=True)

    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
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
    results_pub = rospy.Publisher ('/results', MarkerArray, queue_size=10)
    tracking_img_pub = rospy.Publisher ('/tracking_img', Image, queue_size=10)
    mask_img_pub = rospy.Publisher('/mask', Image, queue_size=10)

    opencv_mask_sub = rospy.Subscriber('/mask_with_occlusion', Image, update_occlusion_mask)

    ts = message_filters.TimeSynchronizer([rgb_sub, pc_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()