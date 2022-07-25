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
import std_msgs.msg

def to_rad(deg):
	return float(deg)/180.0*pi

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('traj_pub_node', anonymous=True)

robot = moveit_commander.RobotCommander()

scene = moveit_commander.PlanningSceneInterface()
group_name = "manipulator"
move_group = moveit_commander.MoveGroupCommander(group_name)

# important - otherwise could cause error
move_group.set_start_state_to_current_state()
move_group.set_max_acceleration_scaling_factor(1)  # ranges from 0 to 1

###########################################################################################
# # joint values

# # We can get the joint values from the group and adjust some of the values:
# joint_goal = move_group.get_current_joint_values()

# joint_goal[0] = 0
# joint_goal[1] = pi/6
# joint_goal[2] = pi/6
# joint_goal[3] = -pi/6
# joint_goal[4] = pi/6
# joint_goal[5] = 0

# # The go command can be called with joint values, poses, or without any
# # parameters if you have already set the pose or joint target for the group
# move_group.go(joint_goal, wait=True)

# # Calling ``stop()`` ensures that there is no residual movement
# move_group.stop()

# ###########################################################################################
# # 3D coord

# pose_goal = geometry_msgs.msg.Pose()
# pose_goal.orientation.w = 0.0
# pose_goal.position.x = 0.4
# pose_goal.position.y = 0.4
# pose_goal.position.z = 0.4

# move_group.set_pose_target(pose_goal)

# move_group.go(wait=True)
# move_group.stop()

###########################################################################################
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