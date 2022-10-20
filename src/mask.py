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

def callback (rgb, depth, pc):

    proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                            [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                            [             0.0,              0.0,               1.0, 0.0]])

    # process rgb image
    cur_image = ros_numpy.numpify(rgb)
    # cur_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_RGB2HSV)

    # color thresholding
    # --- blue ---
    lower = (98, 100, 100)
    upper = (130, 255, 255)
    mask = cv2.inRange(hsv_image, lower, upper)

    # --- green ---
    lower = (85, 130, 60)
    upper = (95, 255, 255)
    mask_green = cv2.inRange(hsv_image, lower, upper).astype('uint8')

    # bmask = mask.copy() # for checking visibility, max = 255
    # mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR).astype('uint8')

    # test
    mask = cv2.bitwise_or(mask.copy(), mask_green.copy()) # mask_green.copy()
    mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2RGB)

    # blob detection
    params = cv2.SimpleBlobDetector_Params()
    # Filter by Color
    params.filterByColor = False
    # Filter by Area.
    params.filterByArea = True
    # Filter by Circularity
    params.filterByCircularity = False
    # Filter by Inerita
    params.filterByInertia = True
    # Filter by Convexity
    params.filterByConvexity = False

    # # Create a detector with the parameters
    # detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(mask)

    # # Find blob centers in the image coordinates
    # blob_image_center = []
    # num_blobs = len(keypoints)
    # for i in range(num_blobs):
    #     blob_image_center.append((keypoints[i].pt[0],keypoints[i].pt[1]))

    # blob_image_center = np.array(blob_image_center)

    # us = blob_image_center[:, 0].astype(int)
    # vs = blob_image_center[:, 1].astype(int)

    # tracking_img = cur_image.copy()
    # for i in range (len(blob_image_center)):
    #     # draw circle
    #     uv = (us[i], vs[i])
    #     cv2.circle(tracking_img, uv, 5, (255, 150, 0), -1)

        # # draw line
        # if i != len(blob_image_center)-1:
        #     cv2.line(tracking_img, uv, (us[i+1], vs[i+1]), (255, 150, 0), 2)
    
    # tracking_img_msg = ros_numpy.msgify(Image, tracking_img, 'rgb8')
    # tracking_img_pub.publish(tracking_img_msg)

    # publish mask
    mask_img_msg = ros_numpy.msgify(Image, mask, 'rgb8')
    mask_img_pub.publish(mask_img_msg)


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