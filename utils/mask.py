#!/usr/bin/env python3

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

def callback (rgb, pc):

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

    # color thresholding
    # --- wire blue ---
    # lower = (98, 100, 100)
    # upper = (130, 255, 255)
    # --- rope blue ---
    lower = (90, 60, 40)
    upper = (130, 255, 255)
    # --- background green ---
    # lower = (0, 0, 0)
    # upper = (220, 255, 210)
    mask = cv2.inRange(hsv_image, lower, upper)

    # --- wire green ---
    # lower = (85, 130, 60)
    # upper = (95, 255, 255)
    # --- tape green ---
    # lower = (40, 110, 60)
    # upper = (85, 255, 255)
    # --- tape red ---
    lower = (130, 60, 40)
    upper = (255, 255, 255)
    mask_red_1 = cv2.inRange(hsv_image, lower, upper).astype('uint8')
    lower = (0, 60, 40)
    upper = (10, 255, 255)
    mask_red_2 = cv2.inRange(hsv_image, lower, upper).astype('uint8')
    mask_red = cv2.bitwise_or(mask_red_1.copy(), mask_red_2.copy())

    # bmask = mask.copy() # for checking visibility, max = 255
    # mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR).astype('uint8')

    # test
    mask = cv2.bitwise_or(mask.copy(), mask_red.copy())
    bmask = mask.copy()
    mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2RGB)

    # test distance transform
    bmask_transformed = cv2.distanceTransform(255-bmask, cv2.DIST_L2, cv2.DIST_MASK_3)
    print(np.amax(bmask_transformed))
    print(bmask_transformed)
    bmask_transformed[bmask_transformed > 255] = 255
    bmask_transformed_rgb = cv2.cvtColor(bmask_transformed.copy().astype('uint8'), cv2.COLOR_GRAY2RGB)
    # cv2.imshow('frame', bmask_transformed)
    # cv2.waitKey(3)

    # blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByArea = True
    params.filterByCircularity = False
    params.filterByInertia = True
    params.filterByConvexity = False

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask)

    # Find blob centers in the image coordinates
    blob_image_center = []
    nodes = []
    num_blobs = len(keypoints)
    tracking_img = cur_image.copy()
    for i in range(num_blobs):
        blob_image_center.append((keypoints[i].pt[0],keypoints[i].pt[1]))
        nodes.append(cur_pc[int(keypoints[i].pt[1]), int(keypoints[i].pt[0])])
        # draw image
        uv = (int(keypoints[i].pt[0]), int(keypoints[i].pt[1]))
        cv2.circle(tracking_img, uv, 5, (255, 150, 0), -1)

    # # add color
    # nodes_rgba = struct.unpack('I', struct.pack('BBBB', 0, 0, 0, 255))[0]
    # nodes_rgba_arr = np.full((len(nodes), 1), nodes_rgba)
    # nodes_colored = np.hstack((nodes, nodes_rgba_arr)).astype('O')
    # nodes_colored[:, 3] = nodes_colored[:, 3].astype(int)
    # header.stamp = rospy.Time.now()
    # converted_nodes = pcl2.create_cloud(header, fields, nodes_colored)
    # nodes_pub.publish(converted_nodes)
    
    tracking_img_msg = ros_numpy.msgify(Image, tracking_img, 'rgb8')
    tracking_img_pub.publish(tracking_img_msg)

    # publish mask
    # mask_img_msg = ros_numpy.msgify(Image, mask, 'rgb8')
    mask_img_msg = ros_numpy.msgify(Image, bmask_transformed_rgb, 'rgb8')
    mask_img_pub.publish(mask_img_msg)


if __name__=='__main__':
    rospy.init_node('test', anonymous=True)

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
    init_nodes_pub = rospy.Publisher ('/init_nodes', PointCloud2, queue_size=10)
    nodes_pub = rospy.Publisher ('/nodes', PointCloud2, queue_size=10)
    guide_nodes_pub = rospy.Publisher ('/guide_nodes', PointCloud2, queue_size=10)
    tracking_img_pub = rospy.Publisher ('/tracking_img', Image, queue_size=10)
    mask_img_pub = rospy.Publisher('/mask', Image, queue_size=10)

    ts = message_filters.TimeSynchronizer([rgb_sub, pc_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()