#!/usr/bin/env python

import os
os.environ["ROS_NAMESPACE"] = "/rob1"

import sys
import moveit_commander
import moveit_msgs.msg
import ast
import rospy
from rospy import Duration
import std_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryActionGoal
import time
from numpy import pi

def functional():

    moveit_commander.roscpp_initialize(sys.argv)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    move_group.set_start_state_to_current_state()
    move_group.set_max_acceleration_scaling_factor(0.2)

    pub_rob1 = rospy.Publisher('/rob1/joint_trajectory_action/goal',
    FollowJointTrajectoryActionGoal, queue_size=1)
    rospy.init_node('traj_maker', anonymous=True)
    time.sleep(1)

    #  making message
    message_rob1 = FollowJointTrajectoryActionGoal()

    #  required headers
    header_rob1 = std_msgs.msg.Header(stamp=rospy.Time.now())
    message_rob1.goal.trajectory.header = header_rob1
    message_rob1.header = header_rob1

    #  adding in joints
    joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', \
                  'joint_5', 'joint_6']
    message_rob1.goal.trajectory.joint_names = joint_names

    joint_goal = move_group.get_current_joint_values()

    # joint_goal[0] = 0
    # joint_goal[1] = pi/4
    # joint_goal[2] = 0
    # joint_goal[3] = -pi/6
    # joint_goal[4] = pi/6
    # joint_goal[5] = 0

    joint_goal[0] = 0
    joint_goal[1] = 0
    joint_goal[2] = 0
    joint_goal[3] = 0
    joint_goal[4] = 0
    joint_goal[5] = 0

    plan = move_group.plan(joint_goal)
    traj = plan.joint_trajectory.points
    message_rob1.goal.trajectory.points = traj
  
    #  publishing to ROS node
    pub_rob1.publish(message_rob1)
               

if __name__ == '__main__':
    try:
        functional()
    except rospy.ROSInterruptException:
        pass
