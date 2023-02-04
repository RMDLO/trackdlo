#!/usr/bin/env python3

import rospy
import ros_numpy
from ros_numpy import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pcl2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

import struct
import cv2
import numpy as np

import time
import yaml
from os.path import dirname, abspath, join

import message_filters
import open3d as o3d
from scipy import ndimage
from scipy.spatial.transform import Rotation as R


class TrackDLO:
    """
    Performs deformable linear object tracking with motion coherence
    """

    def __init__(self):
        self.proj_matrix = np.array([[918.359130859375, 0.0, 645.8908081054688, 0.0],
                                    [0.0, 916.265869140625, 354.02392578125, 0.0],
                                    [0.0, 0.0, 1.0, 0.0]])
        self.occlusion_mask_rgb = None
        self.initialized = False
        self.read_params = False
        self.init_nodes = []
        self.nodes = []
        self.sigma2 = 0
        self.total_len = 0
        self.geodesic_coord = []

        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.pc_sub = message_filters.Subscriber('/camera/depth/color/points', PointCloud2)
        self.opencv_mask_sub = rospy.Subscriber('/mask_with_occlusion', Image, self.update_occlusion_mask)

        self.pc_pub = rospy.Publisher ('/pts', PointCloud2, queue_size=10)
        self.results_pub = rospy.Publisher ('/results', MarkerArray, queue_size=10)
        self.track_pc_pub = rospy.Publisher('/results_pc', PointCloud2, queue_size=10)
        self.guide_nodes_pub = rospy.Publisher ('/guide_nodes', MarkerArray, queue_size=10)
        self.tracking_img_pub = rospy.Publisher ('/tracking_img', Image, queue_size=10)
        self.mask_img_pub = rospy.Publisher('/mask', Image, queue_size=10)

        self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.pc_sub], 10)
        self.ts.registerCallback(self.callback)

    def callback(self, rgb, pc):
        # header
        head = Header()
        head.stamp = rgb.header.stamp
        head.frame_id = 'camera_color_optical_frame'
        
        if not self.read_params:
            setting_path = join(dirname(dirname(abspath(__file__))), "config/TrackDLO_params.yaml")
            with open(setting_path, 'r') as file:
                self.params = yaml.safe_load(file)
            self.read_params = True

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
        if self.occlusion_mask_rgb is None:
            self.occlusion_mask_rgb = np.ones(cur_image.shape).astype('uint8')*255
        occlusion_mask = cv2.cvtColor(self.occlusion_mask_rgb.copy(), cv2.COLOR_RGB2GRAY)

        if not self.params["initialization_params"]["using_rope_with_markers"]:
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
        mask_img_msg.header = head
        self.mask_img_pub.publish(mask_img_msg)

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

        rospy.loginfo("Downsampled point cloud size: " + str(len(filtered_pc)))

        # add color
        pc_rgba = struct.unpack('I', struct.pack('BBBB', 255, 40, 40, 255))[0]
        pc_rgba_arr = np.full((len(filtered_pc), 1), pc_rgba)
        filtered_pc_colored = np.hstack((filtered_pc, pc_rgba_arr)).astype('O')
        filtered_pc_colored[:, 3] = filtered_pc_colored[:, 3].astype(int)

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('rgba', 12, PointField.UINT32, 1)]
        converted_points = pcl2.create_cloud(header=head,
                                            fields=fields,
                                            points=filtered_pc_colored)
        self.pc_pub.publish(converted_points)

        rospy.logwarn('callback before initialized: ' + str((time.time() - cur_time_cb)*1000) + ' ms')

        # register nodes
        if not self.initialized:

            self.init_nodes, self.sigma2 = self.register(filtered_pc, self.params["initialization_params"]["num_of_nodes"], mu=self.params["initialization_params"]["mu"], max_iter=self.params["initialization_params"]["max_iter"])
            self.init_nodes = self.sort_pts(self.init_nodes)

            self.nodes = self.init_nodes.copy()

            # compute preset coord and total len. one time action
            seg_dis = np.sqrt(np.sum(np.square(np.diff(self.init_nodes, axis=0)), axis=1))
            self.geodesic_coord = []
            last_pt = 0
            self.geodesic_coord.append(last_pt)
            for i in range (1, len(self.init_nodes)):
                last_pt += seg_dis[i-1]
                self.geodesic_coord.append(last_pt)
            self.geodesic_coord = np.array(self.geodesic_coord)
            self.total_len = np.sum(np.sqrt(np.sum(np.square(np.diff(self.init_nodes, axis=0)), axis=1)))

            self.initialized = True

        # cpd
        if self.initialized:
            # determined which nodes are occluded from mask information
            mask_dis_threshold = self.params["initialization_params"]["mask_dis_threshold"]
            # projection
            init_nodes_h = np.hstack((self.init_nodes, np.ones((len(self.init_nodes), 1))))
            # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
            image_coords = np.matmul(self.proj_matrix, init_nodes_h.T).T
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
            guide_nodes, self.nodes, self.sigma2 = self.tracking_step(filtered_pc, self.nodes, self.sigma2, self.geodesic_coord, bmask_transformed, mask_dis_threshold)
            rospy.logwarn('tracking_step total: ' + str((time.time() - cur_time)*1000) + ' ms')

            self.init_nodes = self.nodes.copy()

            results = self.ndarray2MarkerArray(self.nodes, [255, 150, 0, 0.75], [0, 255, 0, 0.75], head)
            guide_nodes_results = self.ndarray2MarkerArray(guide_nodes, [0, 0, 0, 0.5], [0, 0, 1, 0.5], head)
            self.results_pub.publish(results)
            self.guide_nodes_pub.publish(guide_nodes_results)

            if self.params["initialization_params"]["pub_tracking_image"]:
                # project and pub tracking image
                nodes_h = np.hstack((self.nodes, np.ones((len(self.nodes), 1))))

                # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
                image_coords = np.matmul(self.proj_matrix, nodes_h.T).T
                us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
                vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

                cur_image_masked = cv2.bitwise_and(cur_image, self.occlusion_mask_rgb)
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
                tracking_img_msg.header = head
                self.tracking_img_pub.publish(tracking_img_msg)

            rospy.logwarn('callback total: ' + str((time.time() - cur_time_cb)*1000) + ' ms')

    def pt2pt_dis_sq(self, pt1, pt2):
        return np.sum(np.square(pt1 - pt2))

    def pt2pt_dis(self, pt1, pt2):
        return np.sqrt(np.sum(np.square(pt1 - pt2)))

    def update_occlusion_mask(self, data):
        self.occlusion_mask_rgb = ros_numpy.numpify(data)

    # original post: https://stackoverflow.com/a/59204638
    def rotation_matrix_from_vectors(self, vec1, vec2):
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

    def ndarray2MarkerArray(self, Y, node_color, line_color, head):
        results = MarkerArray()
        Y_msg = PointCloud2()
        pc_results_list = []

        for i in range (0, len(Y)):
            cur_node_result = Marker()
            cur_node_result.header = head
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
            cur_line_result.header = head
            cur_line_result.type = Marker.CYLINDER
            cur_line_result.action = Marker.ADD
            cur_line_result.ns = "line_results" + str(i)
            cur_line_result.id = i

            cur_line_result.pose.position.x = ((Y[i] + Y[i+1])/2)[0]
            cur_line_result.pose.position.y = ((Y[i] + Y[i+1])/2)[1]
            cur_line_result.pose.position.z = ((Y[i] + Y[i+1])/2)[2]

            rot_matrix = self.rotation_matrix_from_vectors(np.array([0, 0, 1]), (Y[i+1]-Y[i])/self.pt2pt_dis(Y[i+1], Y[i])) 
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
            cur_line_result.scale.z = self.pt2pt_dis(Y[i], Y[i+1])
            cur_line_result.color.r = line_color[0]
            cur_line_result.color.g = line_color[1]
            cur_line_result.color.b = line_color[2]
            cur_line_result.color.a = line_color[3]

            results.markers.append(cur_line_result)
            pt = np.array([Y[i,0],Y[i,1],Y[i,2]]).astype(np.float32)
            pc_results_list.append(pt)
        
        pc = np.vstack(pc_results_list).astype(np.float32).T
        rec_project = np.core.records.fromarrays(pc, 
                                                names='x, y, z',
                                                formats = 'float32, float32, float32')
        Y_msg = point_cloud2.array_to_pointcloud2(rec_project, head.stamp, frame_id='camera_color_optical_frame') # include time stamp matching other time
        self.track_pc_pub.publish(Y_msg)
        
        return results

    def get_estimates(self, Y, s, X, D, M, N, mu):

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

    def register(self, pts, M, mu=0, max_iter=50):

        # initial guess
        X = pts.copy()
        Y = np.vstack((np.arange(0, 0.1, (0.1/M)), np.zeros(M), np.zeros(M))).T
        if len(pts[0]) == 2:
            Y = np.vstack((np.arange(0, 0.1, (0.1/M)), np.zeros(M))).T
        s = 1
        N = len(pts)
        D = len(pts[0])

        prev_Y, prev_s = Y, s
        new_Y, new_s = self.get_estimates(prev_Y, prev_s, X, D, M, N, mu)
        
        for it in range (max_iter):
            prev_Y, prev_s = new_Y, new_s
            new_Y, new_s = self.get_estimates(prev_Y, prev_s, X, D, M, N, mu)

        return new_Y, new_s

    def sort_pts(self, Y_0):
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

    def get_nearest_indices(self, k, Y, idx):
        '''
        Finds nearest neighbors assuming Y is sorted. Finds the nearest k neighbors to
        both the left and the right of the node in the geodesic (total: 2k neighbors)
        '''
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

    def calc_LLE_weights(self, k, X):
        W = np.zeros((len(X), len(X)))
        for i in range (0, len(X)):
            indices = self.get_nearest_indices(int(k/2), X, i)
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

    def ecpd_lle(self,
                X_orig,                      # input point cloud
                Y_0,                         # input nodes
                sigma2_0,                    # initial variance
                beta,                        # MCT kernel strength
                alpha,                       # MCT overall strength
                gamma,                       # LLE strength
                mu,                          # noise
                max_iter = 30,               # how many iterations EM will run
                tol = 0.00001,               # when to terminate the optimization process
                include_lle = True, 
                use_geodesic = False, 
                use_prev_sigma2 = False, 
                use_ecpd = False, 
                correspondence_priors = None,
                omega = None,                 # ecpd strength. DO NOT go lower than 1e-6
                kernel = 'Gaussian',          # Gaussian, Laplacian, 1st order, 2nd order
                occluded_nodes = None,
                k_vis = 0,
                bmask_transformed = None): 

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
                G = 1/(beta**3)*np.sqrt(np.pi) * np.exp(-2*np.sqrt(2)*converted_node_dis/beta) * (2*converted_node_dis_sq + 3*beta/2*np.sqrt(2)*converted_node_dis + 3*beta**2/4) 
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
                G = 1/(beta**3)*np.sqrt(np.pi) * np.exp(-2*np.sqrt(2)*converted_node_dis/beta) * (2*converted_node_dis_sq + 3*beta/2*np.sqrt(2)*converted_node_dis + 3*beta**2/4) 
            else:
                G = np.exp(-converted_node_dis_sq / (2 * beta**2))
        
        Y = Y_0.copy()

        # initialize sigma2
        if not use_prev_sigma2:
            (N, D) = X.shape
            (M, _) = Y.shape
            diff = X[None, :, :] - Y[:, None, :]
            err = diff ** 2
            self.sigma2 = np.sum(err) / (D * M * N)
        else:
            self.sigma2 = sigma2_0

        # get the LLE matrix
        L = self.calc_LLE_weights(2, Y_0)
        H = np.matmul((np.identity(M) - L).T, np.identity(M) - L)

        if correspondence_priors is not None and len(correspondence_priors) != 0:
            additional_pc = correspondence_priors[:, 1:4]
            X = np.vstack((additional_pc, X))

        N = len(X)
        
        # loop until convergence or max_iter reached
        for it in range (0, max_iter):

            # faster P computation
            pts_dis_sq = np.sum((X[None, :, :] - Y[:, None, :]) ** 2, axis=2)
            c = (2 * np.pi * self.sigma2) ** (D / 2)
            c = c * mu / (1 - mu)
            c = c * M / N
            P = np.exp(-pts_dis_sq / (2 * self.sigma2))
            P_stored = P.copy()
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

                P = np.exp(-np.square(geodesic_dists) / (2 * self.sigma2))

            else:
                P = P_stored.copy()

            # use cdcpd's pvis
            if (occluded_nodes is not None) and (len(occluded_nodes) != 0):
                self.nodes = correspondence_priors[:, 1:4]
                nodes_h = np.hstack((self.nodes, np.ones((len(self.nodes), 1))))
                # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
                image_coords = np.matmul(self.proj_matrix, nodes_h.T).T
                xs = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
                ys = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

                # modified probability distribution
                P_vis = np.ones((M, N))
                total_P_vis = 0

                # for loop for now, will change later
                for i in range (len(image_coords)):
                    x = xs[i]
                    y = ys[i]
                    pixel_dist = bmask_transformed[y, x]
                    P_vis_i = np.exp(-k_vis*pixel_dist)
                    total_P_vis += P_vis_i
                    P_vis[i] *= P_vis_i
                
                # normalize P_vis
                P_vis = P_vis / total_P_vis

                # modify P
                P = P_vis * P

                den = np.sum(P, axis=0)
                den = np.tile(den, (M, 1))
                den[den == 0] = np.finfo(float).eps
                c = (2 * np.pi * self.sigma2) ** (D / 2) * mu / (1 - mu) / N
                den += c
                P = np.divide(P, den)

            else:
                den = np.sum(P, axis=0)
                den = np.tile(den, (M, 1))
                den[den == 0] = np.finfo(float).eps
                c = (2 * np.pi * self.sigma2) ** (D / 2)
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
                    # correspondence priors: index, x, y, z
                    for i in range (len(correspondence_priors)):
                        index = correspondence_priors[i, 0]
                        P_tilde[int(index), i] = 1

                    P_tilde_1 = np.sum(P_tilde, axis=1)
                    P_tilde_X = np.matmul(P_tilde, X)

                    A_matrix = np.matmul(np.diag(P1), G) + alpha * self.sigma2 * np.identity(M) + self.sigma2 * gamma * np.matmul(H, G) + self.sigma2 / omega * np.matmul(np.diag(P_tilde_1), G)
                    B_matrix = PX - np.matmul(np.diag(P1) + self.sigma2*gamma*H, Y_0) + self.sigma2 / omega * (P_tilde_X - np.matmul(np.diag(P_tilde_1) + self.sigma2*gamma*H, Y_0))
                else:
                    A_matrix = np.matmul(np.diag(P1), G) + alpha * self.sigma2 * np.identity(M) + self.sigma2 * gamma * np.matmul(H, G)
                    B_matrix = PX - np.matmul(np.diag(P1) + self.sigma2*gamma*H, Y_0)
            else:
                if use_ecpd:
                    P_tilde = np.zeros((M, N))
                    pt_node_correspondence = np.argmax(P, axis=0)
                    
                    for node_num in range (0, M):
                        node_num_pts_indices = np.where(pt_node_correspondence == node_num)
                        P_tilde[node_num, node_num_pts_indices] = 1

                    P_tilde_1 = np.sum(P_tilde, axis=1)
                    P_tilde_X = np.matmul(P_tilde, X)

                    A_matrix = np.matmul(np.diag(P1), G) + alpha * self.sigma2 * np.identity(M) + self.sigma2 / omega * np.matmul(np.diag(P_tilde_1), G)
                    B_matrix = PX - np.matmul(np.diag(P1), Y_0) + self.sigma2 / omega * (P_tilde_X - np.matmul(np.diag(P_tilde_1), Y_0))
                else:
                    A_matrix = np.matmul(np.diag(P1), G) + alpha * self.sigma2 * np.identity(M)
                    B_matrix = PX - np.matmul(np.diag(P1), Y_0)

            W = np.linalg.solve(A_matrix, B_matrix)

            T = Y_0 + np.matmul(G, W)
            trXtdPt1X = np.trace(np.matmul(np.matmul(X.T, np.diag(Pt1)), X))
            trPXtT = np.trace(np.matmul(PX.T, T))
            trTtdP1T = np.trace(np.matmul(np.matmul(T.T, np.diag(P1)), T))

            self.sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D)

            # update Y
            if self.pt2pt_dis_sq(Y, Y_0 + np.matmul(G, W)) < tol:
                Y = Y_0 + np.matmul(G, W)
                rospy.loginfo('Iterations until convergence: ' + str(it) + '. Kernel: ' + kernel)
                break
            else:
                Y = Y_0 + np.matmul(G, W)

            if it == max_iter - 1:
                # print error messages if optimization did not compile
                rospy.logerr('Optimization did not converge! ' + 'Kernel: ' + kernel)

        return Y, self.sigma2

    def tracking_step(self, X_orig, Y_0, sigma2_0, geodesic_coord, bmask_transformed, mask_dist_threshold):
        # log time
        cur_time = time.time()

        # projection
        nodes_h = np.hstack((Y_0, np.ones((len(Y_0), 1))))
        image_coords = np.matmul(self.proj_matrix, nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

        us = np.where(us >= 1280, 1279, us)
        vs = np.where(vs >= 720, 719, vs)

        uvs = np.vstack((vs, us)).T
        uvs_t = tuple(map(tuple, uvs.T))
        vis = bmask_transformed[uvs_t]
        occluded_nodes = np.where(vis > mask_dist_threshold)[0]
        visible_nodes = np.where(vis <= mask_dist_threshold)[0]

        guide_nodes = Y_0[visible_nodes]
        guide_nodes, _ = self.ecpd_lle(X_orig, guide_nodes, 0, 10000, 1, 1, 0.05, 50, 0.00001, True, True, False, False)
        correspondence_priors = np.vstack((visible_nodes, guide_nodes.T)).T

        Y, self.sigma2 = self.ecpd_lle(X_orig, Y_0, sigma2_0, 10, 1, 2, 0.05, 50, 0.00001, True, True, True, True, correspondence_priors, 0.000005, "1st order", occluded_nodes, 0.015, bmask_transformed)

        rospy.logwarn('tracking_step registration: ' + str((time.time() - cur_time)*1000) + ' ms')

        return correspondence_priors[:, 1:4], Y, self.sigma2

if __name__=='__main__':
    rospy.init_node('trackdlo')
    t = TrackDLO()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")