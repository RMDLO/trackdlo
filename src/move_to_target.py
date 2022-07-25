#!/usr/bin/env python

import os
# os.environ["ROS_NAMESPACE"] = "/rob1"

import time
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryActionGoal
from std_msgs.msg import String
import tf
from scipy.spatial.transform import Rotation as R
import numpy as np

def to_rad(deg):
	return float(deg)/180.0*pi

uv_arr = []

def update_uv(data):
	global uv_arr
	string = data.data
	string_arr = string.split(' ')
	uv_arr.append(float(string_arr[0]))
	uv_arr.append(float(string_arr[1]))


def tf2mat(trans, quat):
	rot_matrix = R.from_quat(quat).as_dcm()
	t = np.zeros((4, 4))
	t[0:3, 0:3] = rot_matrix
	t[0:3, 3] = trans
	t[3, 3] = 1.0

	return t


def get_xy_coord(proj_matrix, R_ew, uv):
	const_matrix = np.matmul(proj_matrix, R_ew)
	A = np.zeros((3, 3))
	A[:, 0:2] = const_matrix[:, 0:2]
	A[0, 2] = -uv[0]
	A[1, 2] = -uv[1]
	A[2, 2] = -1
	B = -const_matrix[:, 3]
	sol = np.matmul(np.linalg.inv(A), B)

	return sol[0], sol[1], sol[2]


proj_matrix = np.array([[1377.5386962890625,                0.0, 968.836181640625, 0.0], \
					    [               0.0, 1374.3988037109375, 531.035888671875, 0.0], \
					    [               0.0,                0.0,              1.0, 0.0]])

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_robot', anonymous=True)

robot = moveit_commander.RobotCommander()

scene = moveit_commander.PlanningSceneInterface()
group_name = "manipulator"
move_group = moveit_commander.MoveGroupCommander(group_name)

# important - otherwise could cause error
move_group.set_start_state_to_current_state()
move_group.set_max_acceleration_scaling_factor(0.5)  # ranges from 0 to 1

listener = tf.TransformListener()

# going back to starting point

joint_goal = move_group.get_current_joint_values()

joint_goal[0] = 0
joint_goal[1] = 0
joint_goal[2] = 0
joint_goal[3] = 0
joint_goal[4] = 0
joint_goal[5] = 0

move_group.go(joint_goal, wait=True)
move_group.stop()

time.sleep(2)

trans, quat = listener.lookupTransform('camera_color_optical_frame', 'base_link', rospy.Time())
R_ew = tf2mat(trans, quat)
rospy.Subscriber("/corner_uv", String, update_uv)
time.sleep(0.2)

if len(uv_arr) == 0:
	print('Could not get updated uv!')
	exit()

time.sleep(1)

target_x, target_y, _ = get_xy_coord(proj_matrix, R_ew, uv_arr)
print(target_x, target_y, _)

target_z = 0.0005

# ###########################################################################################
# 3D coord
pose_goal = geometry_msgs.msg.Pose()
pose_goal.orientation.w = 0.0
pose_goal.orientation.x = -0.707
pose_goal.orientation.y = 0.707
pose_goal.orientation.z = 0.0

pose_goal.position.x = target_x
pose_goal.position.y = target_y
pose_goal.position.z = target_z

move_group.set_pose_target(pose_goal)

move_group.go(wait=True)
move_group.stop()
###########################################################################################
