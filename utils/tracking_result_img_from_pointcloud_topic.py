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
import open3d as o3d
from scipy import ndimage
from scipy import interpolate

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

def callback (pc):
    global cur_image
    global bmask
    global mask

    proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                            [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                            [             0.0,              0.0,               1.0, 0.0]])

    # process point cloud
    pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc)
    result_pc = ros_numpy.point_cloud2.get_xyz_points(pc_data)
    nodes = result_pc.copy()

    # determined which nodes are occluded from mask information
    mask_dis_threshold = 10
    # projection
    nodes_h = np.hstack((nodes, np.ones((len(nodes), 1))))
    # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
    image_coords = np.matmul(proj_matrix, nodes_h.T).T
    us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
    vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

    # limit the range of calculated image coordinates
    us = np.where(us >= 1280, 1279, us)
    vs = np.where(vs >= 720, 719, vs)

    uvs = np.vstack((vs, us)).T
    uvs_t = tuple(map(tuple, uvs.T))

    # invert bmask for distance transform
    bmask_transformed = ndimage.distance_transform_edt(255 - bmask)
    vis = bmask_transformed[uvs_t]

    tracking_img = cur_image.copy()
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

if __name__=='__main__':
    rospy.init_node('cdcpd_image', anonymous=True)

    rospy.Subscriber('/mask', Image, update_mask)
    rospy.Subscriber('/camera/color/image_raw', Image, update_rgb)
    rospy.Subscriber('/cdcpd2_no_gripper_results_pc', PointCloud2, callback)
    tracking_img_pub = rospy.Publisher ('/tracking_img', Image, queue_size=10)

    rospy.spin()