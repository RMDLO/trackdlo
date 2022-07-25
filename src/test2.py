#!/usr/bin/env python

import os
os.environ["ROS_NAMESPACE"] = "/rob1"

import ast
import rospy
from rospy import Duration
import std_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryActionGoal
import time
from numpy import pi

def functional():
    pub_rob1 = rospy.Publisher('/rob1/joint_trajectory_action/goal',
     FollowJointTrajectoryActionGoal, queue_size=1)
    rospy.init_node('traj_maker', anonymous=True)
    time.sleep(4)

    rate = rospy.Rate(0.01)

    print("loop")

    while not rospy.is_shutdown():

           traj_waypoint_1_rob1 = JointTrajectoryPoint()
           traj_waypoint_2_rob1 = JointTrajectoryPoint()

           traj_waypoint_1_rob1.positions = [0,0,0,-pi/6,pi/6,0]
           traj_waypoint_2_rob1.positions = [0,0,0,0,0,0]       

           traj_waypoint_1_rob1.time_from_start.nsecs = 1000000000
           traj_waypoint_2_rob1.time_from_start.nsecs = 2000000000
           
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
           
           #  appending up to 100 points
           # ex. for i in enumerate(len(waypoints)): append(waypoints[i])
           message_rob1.goal.trajectory.points.append(traj_waypoint_1_rob1)
           message_rob1.goal.trajectory.points.append(traj_waypoint_2_rob1)
          
           #  publishing to ROS node
           pub_rob1.publish(message_rob1)
           print(message_rob1)
         
           rate.sleep()
           
           if rospy.is_shutdown():
               break
           break
               

if __name__ == '__main__':
    try:
        functional()
    except rospy.ROSInterruptException:
        pass
