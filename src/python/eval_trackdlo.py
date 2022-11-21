#!/usr/bin/env python

from tracking_dev_ros import pt2pt_dis_sq, pt2pt_dis, sort_pts, tracking_step, ecpd_lle

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg
import message_filters
import open3d as o3d
from scipy import ndimage

import struct
import time
import cv2
import numpy as np
import math
from scipy import ndimage
import yaml
from os.path import dirname, abspath, join

occlusion_mask_rgb = None
def update_occlusion_mask(data):
	global occlusion_mask_rgb
	occlusion_mask_rgb = ros_numpy.numpify(data)

initialized = False
read_params = False
init_nodes = []
nodes = []
guide_nodes_Y_0 = []
sigma2 = 0
guide_nodes_sigma2_0 = 0
total_len = 0
geodesic_coord = []
last_guide_node_head = None
def callback (rgb, pc):
    global initialized
    global init_nodes, nodes, sigma2
    global total_len, geodesic_coord
    global guide_nodes_Y_0, guide_nodes_sigma2_0
    global params, read_params
    global occlusion_mask_rgb
    global last_guide_node_head

    if not read_params:
        setting_path = join(dirname(dirname(dirname(abspath(__file__)))), "settings/TrackDLO_params.yaml")
        with open(setting_path, 'r') as file:
            params = yaml.safe_load(file)
        read_params = True

    # log time
    cur_time_cb = time.time()
    print('----------')

    proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                            [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                            [             0.0,              0.0,               1.0, 0.0]])

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
    bmask = mask.copy()
    mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2RGB) # should be the mask of the whole wire

    # blob detection
    blob_params = cv2.SimpleBlobDetector_Params()
    blob_params.filterByColor = False
    blob_params.filterByArea = True
    blob_params.filterByCircularity = False
    blob_params.filterByInertia = True
    blob_params.filterByConvexity = False

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(blob_params)
    keypoints = detector.detect(mask_marker)

    # Find blob centers in the image coordinates
    blob_image_center = []
    guide_nodes = []
    num_blobs = len(keypoints)
    tracking_img = cur_image.copy()

    for i in range(num_blobs):
        blob_image_center.append((keypoints[i].pt[0],keypoints[i].pt[1]))
        guide_nodes.append(cur_pc[int(keypoints[i].pt[1]), int(keypoints[i].pt[0])].tolist())

    # sort guide nodes
    if last_guide_node_head is None:
        guide_nodes = np.array(sort_pts(guide_nodes))
        last_guide_node_head = guide_nodes[0]
    else:
        guide_nodes = np.array(sort_pts(guide_nodes))
        if pt2pt_dis(last_guide_node_head, guide_nodes[-1]) < 0.05:
            # need to reverse
            guide_nodes = guide_nodes.tolist()
            guide_nodes.reverse()
            guide_nodes = np.array(guide_nodes)
            last_guide_node_head = guide_nodes[0]

    # remap name because guide_nodes will be re-assigned to something else later
    blob_nodes = guide_nodes.copy()

    # publish mask
    mask_img_msg = ros_numpy.msgify(Image, mask, 'rgb8')
    mask_img_pub.publish(mask_img_msg)

    mask = (mask/255).astype(int)

    filtered_pc = cur_pc*mask
    filtered_pc = filtered_pc[((filtered_pc[:, :, 0] != 0) | (filtered_pc[:, :, 1] != 0) | (filtered_pc[:, :, 2] != 0))]
    # filtered_pc = filtered_pc[filtered_pc[:, 2] < 0.705]
    # filtered_pc = filtered_pc[filtered_pc[:, 2] > 0.4]

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
        # guide_nodes = np.array(sort_pts(guide_nodes))
        # use ecpd to get the variance
        # correspondence priors: [index, x, y, z]
        temp = np.arange(0, params["initialization_params"]["num_of_nodes"], 1)
        correspondence_priors = np.vstack((temp, guide_nodes.T)).T
        init_nodes, sigma2 = ecpd_lle (X_orig = filtered_pc,                           # input point cloud
                                       Y_0 = guide_nodes,                         # input nodes
                                       beta = 0.5,                        # MCT kernel strength
                                       alpha = 1,                       # MCT overall strength
                                       gamma = 1,                       # LLE strength
                                       mu = 0.1,                          # noise
                                       max_iter = 30,                    # how many iterations EM will run
                                       tol = 0.00001,                         # when to terminate the optimization process
                                       include_lle = True, 
                                       use_geodesic = False, 
                                       use_prev_sigma2 = False, 
                                       sigma2_0 = None,              # initial variance
                                       use_ecpd = True, 
                                       correspondence_priors = correspondence_priors,
                                       omega = 0.00001,                 # ecpd strength. DO NOT go lower than 1e-6
                                       kernel = 'Gaussian',          # Gaussian, Laplacian, 1st order, 2nd order
                                       occluded_nodes = None)
        
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
        mask_dis_threshold = params["initialization_params"]["mask_dis_threshold"]
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
        guide_nodes, nodes, sigma2, guide_nodes_Y_0, guide_nodes_sigma2_0 = tracking_step(params, filtered_pc, init_nodes, sigma2, geodesic_coord, total_len, bmask, guide_nodes_Y_0, guide_nodes_sigma2_0)
        rospy.logwarn('tracking_step total: ' + str((time.time() - cur_time)*1000) + ' ms')

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
        # nodes_h = np.hstack((guide_nodes, np.ones((len(nodes), 1))))

        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

        # limit the range of calculated image coordinates
        us = np.where(us >= 1280, 1279, us)
        vs = np.where(vs >= 720, 719, vs)

        tracking_img = cur_image.copy()
        # visualize manual occlusion as black block
        tracking_img = ((tracking_img*np.clip(occlusion_mask_rgb/255, 0.5, 1)).astype('uint8')).astype('uint8')

        for i in range (len(image_coords)):
            # draw circle
            uv = (us[i], vs[i])
            if vis[i] < mask_dis_threshold:
                cv2.circle(tracking_img, uv, 5, (0, 255, 0), -1)
            else:
                cv2.circle(tracking_img, uv, 5, (255, 0, 0), -1)

            # draw ground truth points
            uv_gt = (int(keypoints[i].pt[0]), int(keypoints[i].pt[1]))
            cv2.circle(tracking_img, uv_gt, 5, (255, 150, 0), -1)

            # draw line
            if i != len(image_coords)-1:
                if vis[i] < mask_dis_threshold:
                    cv2.line(tracking_img, uv, (us[i+1], vs[i+1]), (0, 255, 0), 2)
                else:
                    cv2.line(tracking_img, uv, (us[i+1], vs[i+1]), (255, 0, 0), 2)
        
        tracking_img_msg = ros_numpy.msgify(Image, tracking_img, 'rgb8')
        tracking_img_pub.publish(tracking_img_msg)

        error = np.sum(np.sqrt(np.sum(np.square(blob_nodes - nodes), axis=1))) / params["initialization_params"]["num_of_nodes"]
        error_pub.publish(error)

        rospy.logwarn('callback total: ' + str((time.time() - cur_time_cb)*1000) + ' ms')

if __name__=='__main__':
    rospy.init_node('eval', anonymous=True)

    # wait 15 seconds for auto exposure adjustment
    # time.sleep(15)

    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    pc_sub = message_filters.Subscriber('/camera/depth/color/points', PointCloud2)
    opencv_mask_sub = rospy.Subscriber('/mask_with_occlusion', Image, update_occlusion_mask)

    # header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'camera_color_optical_frame'
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgba', 12, PointField.UINT32, 1)]
    pc_pub = rospy.Publisher ('/pts', PointCloud2, queue_size=10)
    init_nodes_pub = rospy.Publisher ('/trackdlo/init_nodes', PointCloud2, queue_size=10)
    nodes_pub = rospy.Publisher ('/trackdlo/nodes', PointCloud2, queue_size=10)
    guide_nodes_pub = rospy.Publisher ('/trackdlo/guide_nodes', PointCloud2, queue_size=10)
    tracking_img_pub = rospy.Publisher ('/trackdlo/tracking_img', Image, queue_size=10)
    mask_img_pub = rospy.Publisher('/trackdlo/mask', Image, queue_size=10)
    error_pub = rospy.Publisher('/trackdlo/error', std_msgs.msg.Float32, queue_size=10)

    ts = message_filters.TimeSynchronizer([rgb_sub, pc_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()