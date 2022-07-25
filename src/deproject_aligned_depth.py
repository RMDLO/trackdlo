#!/usr/bin/env python

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg
from sensor_msgs.msg import JointState

import math
import struct
import time
import cv2
import numpy as np
import pyrealsense2 as rs

import time
import tf
import csv
import pickle as pkl
from scipy.spatial.transform import Rotation as R
import os

w, h =  424, 240
cx, cy =  213.96360778808594, 118.00798034667969
fx, fy =  306.11968994140625, 305.4219665527344

cur_depth_arr = []

def update_depth(data):
	global cur_depth_arr
	cur_depth_arr = ros_numpy.numpify(data)

# copied from stackoverflow
def indices_array_generic(m,n):
    r0 = np.arange(m) 
    r1 = np.arange(n)
    out = np.empty((m,n,2),dtype=int)
    out[:,:,0] = r0[:,None]
    out[:,:,1] = r1
    return out

def deproject():
	rate = rospy.Rate(1)
	depth_pub = rospy.Publisher ('/points1', PointCloud2, queue_size=1)

	# header
	header = std_msgs.msg.Header()
	header.stamp = rospy.Time.now()
	header.frame_id = 'camera_color_optical_frame'

	fields = [PointField('x', 0, PointField.FLOAT32, 1),
	          PointField('y', 4, PointField.FLOAT32, 1),
	          PointField('z', 8, PointField.FLOAT32, 1)]

	while not rospy.is_shutdown():
		# rospy.Subscriber("/camera/depth/image_rect_raw", Image, update_depth)
		rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, update_depth)
		time.sleep(0.2)

		if len(cur_depth_arr) == 0:
			print("unable to get depth map!")

		else:
			# convert from mm to m
			Z = cur_depth_arr.copy()/1000.0

			X = np.arange(0, w)
			X = np.full((h, w), X)
			X = (X - cx) * Z / fx

			Y = np.arange(0, h)
			Y = np.full((w, h), Y)
			Y = Y.T
			Y = (Y - cy) * Z / fy

			X = np.expand_dims(X, axis=2)
			Y = np.expand_dims(Y, axis=2)
			Z = np.expand_dims(Z, axis=2)

			pcl = np.append(X, Y, axis=2)
			pcl = np.append(pcl, Z, axis=2)
			pcl = pcl.copy().reshape(-1, pcl.shape[-1])

			header.stamp = rospy.Time.now()
			converted_points = pcl2.create_cloud(header, fields, pcl)
			depth_pub.publish(converted_points)

		rate.sleep()

if __name__ == '__main__':
	rospy.init_node('depth', anonymous=True)
	deproject()