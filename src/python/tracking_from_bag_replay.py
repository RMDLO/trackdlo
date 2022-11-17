#!/usr/bin/env python

from tracking_dev_ros import pt2pt_dis_sq, pt2pt_dis, register, sort_pts_mst, tracking_step

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
from scipy import ndimage

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

initialized = False
init_nodes = []
nodes = []
guide_nodes_Y_0 = []
sigma2 = 0
guide_nodes_sigma2_0 = 0
total_len = 0
geodesic_coord = []
def callback (pc):
    global initialized
    global init_nodes
    global nodes
    global sigma2
    global total_len
    global geodesic_coord
    global guide_nodes_Y_0
    global guide_nodes_sigma2_0
    # log time
    cur_time_cb = time.time()
    print('----------')

    proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                            [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                            [             0.0,              0.0,               1.0, 0.0]])

    # process point cloud
    pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc)
    filtered_pc = ros_numpy.point_cloud2.get_xyz_points(pc_data)

    # register nodes
    if not initialized:
        init_nodes, sigma2 = register(filtered_pc, 30, mu=0.05, max_iter=100)
        init_nodes = sort_pts_mst(init_nodes)

        guide_nodes_Y_0 = init_nodes.copy()
        guide_nodes_sigma2_0 = sigma2

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

        # limit the range of calculated image coordinates
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
        guide_nodes, nodes, sigma2, guide_nodes_Y_0, guide_nodes_sigma2_0 = tracking_step(filtered_pc, init_nodes, sigma2, geodesic_coord, total_len, bmask, guide_nodes_Y_0, guide_nodes_sigma2_0)
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
        # nodes_h = np.hstack((guide_nodes, np.ones((len(nodes), 1)))) # TEMP

        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

        # limit the range of calculated image coordinates
        us = np.where(us >= 1280, 1279, us)
        vs = np.where(vs >= 720, 719, vs)

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

        rospy.logwarn('callback total: ' + str((time.time() - cur_time_cb)*1000) + ' ms')

if __name__=='__main__':
    rospy.init_node('test', anonymous=True)

    rospy.Subscriber('/camera/color/image_raw', Image, update_rgb)
    rospy.Subscriber('/mask', Image, update_mask)
    filtered_pc_sub = rospy.Subscriber('/pts', PointCloud2, callback)

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
    tracking_img_pub = rospy.Publisher ('/trackdlo/tracking_img', Image, queue_size=10)
    guide_nodes_pub = rospy.Publisher ('/guide_nodes', PointCloud2, queue_size=10)

    rospy.spin()