#!/usr/bin/env python

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pcl2

import struct
import cv2
import numpy as np
import pyrealsense2 as rs

import time
import tf
import csv
import pickle as pkl
from scipy.spatial.transform import Rotation as R
import os

import sys
import copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryActionGoal

cur_pcl = []

# this gives pcl in the camera's frame
def update_cur_pcl(data):
	global cur_pcl
	pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_array(data)
	cur_pcl = ros_numpy.point_cloud2.get_xyz_points(pcl_arr)

def get_depth_info(data_dir, pkl_name):
	listener = tf.TransformListener()
	pcl_pub = rospy.Publisher ('/points_roi', PointCloud2, queue_size=10)

	rate = rospy.Rate(10)
	time.sleep(0.1)

	trans, quat = listener.lookupTransform('base_link', 'camera_depth_optical_frame', rospy.Time())
	rot_matrix = R.from_quat(quat).as_dcm()
	T = np.zeros((4, 4))
	T[0:3, 0:3] = rot_matrix
	T[0:3, 3] = trans
	T[3, 3] = 1.0

	# moveit stuff
	moveit_commander.roscpp_initialize(sys.argv)
	robot = moveit_commander.RobotCommander()

	scene = moveit_commander.PlanningSceneInterface()
	group_name = "manipulator"
	move_group = moveit_commander.MoveGroupCommander(group_name)

	# important - otherwise could cause error
	move_group.set_start_state_to_current_state()
	move_group.set_max_acceleration_scaling_factor(0.5)
	target_x = 0.40
	target_y = 0.0
	target_z = 0.40
	target_qw = 0.0
	target_qx = -0.707
	target_qy = 0.707
	target_qz = 0.0

	samples = []

	while target_z <= 0.75:

		print("in loop")

		avg_z = 0
		i = 0
		sample = []

		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.orientation.w = target_qw
		pose_goal.orientation.x = target_qx
		pose_goal.orientation.y = target_qy
		pose_goal.orientation.z = target_qz

		pose_goal.position.x = target_x
		pose_goal.position.y = target_y
		pose_goal.position.z = target_z

		move_group.set_pose_target(pose_goal)

		move_group.go(wait=True)
		move_group.stop()

		print("finished executing")

		counter = 0
		failed_attempt = False

		while i < 50 and (not failed_attempt):

			if counter > 20:
				failed_attempt = True
				break

			rospy.Subscriber("/camera/depth/color/points", PointCloud2, update_cur_pcl)

			try:
				if len(cur_pcl) == 0:
					print("Raw point cloud is empty!")
					counter += 1
					continue

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
					print("No point cloud within ROI!")
					counter += 1
					continue

				# converted_pcl = converted_pcl[converted_pcl[:, 1] < 0]

				# prevent picking up inconsistent points
				converted_pcl = converted_pcl[converted_pcl[:, 0] > 0.3]

				avg_z += np.average(converted_pcl[:, 2])
				i += 1

				header = std_msgs.msg.Header()
				header.stamp = rospy.Time.now()
				header.frame_id = 'base_link'

				fields = [PointField('x', 0, PointField.FLOAT32, 1), \
		              	  PointField('y', 4, PointField.FLOAT32, 1), \
		              	  PointField('z', 8, PointField.FLOAT32, 1)]

				pcl = pcl2.create_cloud(header, fields, converted_pcl)
				pcl_pub.publish(pcl)

			except:
				continue

			rate.sleep()

		if not failed_attempt:
			print("pcl average z = " +str(avg_z/50.0))
			sample.append(avg_z/50.0)

			got_transform = False
			while not got_transform:
				try:
					trans, quat = listener.lookupTransform('base_link', 'camera_depth_optical_frame', rospy.Time())
					got_transform = True
				except:
					continue

			sample.append(trans)
			sample.append(quat)
			samples.append(sample)
			target_z += 0.01
		else:
			print("Invalid z position!")
			target_z += 0.01

	f = open(data_dir + pkl_name, "wb")
	pkl.dump(samples, f)
	f.close()


if __name__ == '__main__':
	rospy.init_node('pcl_info', anonymous=True)
	data_dir = "/home/jingyixiang/test/recorded_depth/6_30_22_stick/"
	pkl_name = "depth_data.json"

	try:
		os.listdir(data_dir)
	except:
		print("Invalid data directory!")
		rospy.signal_shutdown('')

	get_depth_info(data_dir, pkl_name)
