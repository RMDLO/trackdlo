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

cur_pcl = []

# this gives pcl in the camera's frame
def update_cur_pcl(data):
	global cur_pcl
	pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_array(data)
	cur_pcl = ros_numpy.point_cloud2.get_xyz_points(pcl_arr)

def get_depth_info():
	listener = tf.TransformListener()
	# pcl_left_pub = rospy.Publisher ('/points_left', PointCloud2, queue_size=1)
	pcl_right_pub = rospy.Publisher ('/points_right', PointCloud2, queue_size=10)

	rate = rospy.Rate(2)
	time.sleep(0.1)

	while not rospy.is_shutdown():

		avg_z = 0
		i = 0

		counter = 0
		failed_attempt = False

		while i < 10:

			if counter > 20:
				failed_attempt = True
				break

			rospy.Subscriber("/camera/depth/color/points", PointCloud2, update_cur_pcl)
			time.sleep(0.1)

			if len(cur_pcl) == 0:
				counter += 1
				print("Raw point cloud is empty!")
				continue

			trans, quat = listener.lookupTransform('base_link', 'camera_depth_optical_frame', rospy.Time())
			rot_matrix = R.from_quat(quat).as_dcm()
			T = np.zeros((4, 4))
			T[0:3, 0:3] = rot_matrix
			T[0:3, 3] = trans
			T[3, 3] = 1.0

			cur_pcl_t = np.hstack((cur_pcl, np.ones((len(cur_pcl), 1))))
			converted_pcl = np.matmul(T, cur_pcl_t.T)[0:3, :].T

			# set roi
			# x
			converted_pcl = converted_pcl[converted_pcl[:, 0] > 0.15]
			converted_pcl = converted_pcl[converted_pcl[:, 0] < 0.5]
			# y
			converted_pcl = converted_pcl[converted_pcl[:, 1] > -0.2]
			converted_pcl = converted_pcl[converted_pcl[:, 1] < 0.2]
			# z
			converted_pcl = converted_pcl[converted_pcl[:, 2] > 0.075]

			if len(converted_pcl) == 0:
				counter += 1
				print("No point cloud within ROI!")
				continue

			# converted_pcl_right = converted_pcl[converted_pcl[:, 1] < 0]
			converted_pcl_right = converted_pcl.copy()
			# prevent picking up inconsistent points
			converted_pcl_right = converted_pcl_right[converted_pcl_right[:, 0] > 0.3]

			# converted_pcl_left = converted_pcl[converted_pcl[:, 1] > 0]

			# print("left pcl average z = " +str(np.average(converted_pcl_left[:, 2])))
			avg_z += np.average(converted_pcl_right[:, 2])

			header = std_msgs.msg.Header()
			header.stamp = rospy.Time.now()
			header.frame_id = 'base_link'

			fields = [PointField('x', 0, PointField.FLOAT32, 1), \
	              	  PointField('y', 4, PointField.FLOAT32, 1), \
	              	  PointField('z', 8, PointField.FLOAT32, 1)]

			# pcl_left = pcl2.create_cloud(header, fields, converted_pcl_left)
			# pcl_left_pub.publish(pcl_left)

			pcl_right = pcl2.create_cloud(header, fields, converted_pcl_right)
			pcl_right_pub.publish(pcl_right)
			i += 1

			rate.sleep()

		if failed_attempt:
			print("Invalid z position!")
		else:
			print("pcl average z = " + str(avg_z/10.0))

if __name__ == '__main__':
	rospy.init_node('pcl_info', anonymous=True)
	get_depth_info()