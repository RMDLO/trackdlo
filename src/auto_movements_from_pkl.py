#!/usr/bin/env python
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
from sensor_msgs.msg import JointState
import pickle as pkl

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('traj_pub_node', anonymous=True)

robot = moveit_commander.RobotCommander()

scene = moveit_commander.PlanningSceneInterface()
group_name = "manipulator"
move_group = moveit_commander.MoveGroupCommander(group_name)

# important - otherwise could cause error
move_group.set_start_state_to_current_state()
move_group.set_max_acceleration_scaling_factor(0.5)

joint_goal = move_group.get_current_joint_values()

print("Moving to zero position...")
joint_goal = [0, 0, 0, 0, 0, 0]
move_group.go(joint_goal, wait=True)
move_group.stop()
print("Ready \n")

# read pkl file to get recorded joint angles
f = open("/home/jingyixiang/test/recorded_depth/6_30_22_stick/params/joint_angles.json", "rb")
pkl_list = pkl.load(f)
f.close()

i = 0

while (i < len(pkl_list)) and (not rospy.is_shutdown()):

	print("======================================================================")
	print("Movement " + str(i + 1) + ", joint goals = \n")
	print(pkl_list[i])
	print(" ")
	key_pressed = raw_input("Press enter to move the arm or press q + enter to exit the program \n")

	if key_pressed == 'q':
		print("Shutting down...")
		rospy.signal_shutdown('')
	else:
		print("Planning...")
		joint_goal = pkl_list[i]
		move_group.go(joint_goal, wait=True)
		move_group.stop()
		print("Execution successful! \n")
		i += 1
