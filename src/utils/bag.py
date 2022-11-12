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

def callback (rgb, pc):

    cur_time = time.time()

    proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                            [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                            [             0.0,              0.0,               1.0, 0.0]])

    # process rgb image
    cur_image = ros_numpy.numpify(rgb)
    hsv_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_RGB2HSV)

    # process point cloud
    pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc)
    cur_pc = ros_numpy.point_cloud2.get_xyz_points(pc_data)
    cur_pc = cur_pc.reshape((720, 1280, 3))

    # color thresholding
    lower = (90, 90, 90)
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

    header.stamp = rospy.Time.now()
    converted_points = pcl2.create_cloud(header, fields, filtered_pc)
    pc_pub.publish(converted_points)

    rospy.logwarn('Total time: ' + str(time.time() - cur_time))

if __name__=='__main__':
    rospy.init_node('record_bag', anonymous=True)

    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    pc_sub = message_filters.Subscriber('/camera/depth/color/points', PointCloud2)

    # header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'camera_color_optical_frame'
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)]
                # PointField('rgba', 12, PointField.UINT32, 1)]
    pc_pub = rospy.Publisher ('/pts', PointCloud2, queue_size=10)
    mask_img_pub = rospy.Publisher('/mask', Image, queue_size=10)

    ts = message_filters.TimeSynchronizer([rgb_sub, pc_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()