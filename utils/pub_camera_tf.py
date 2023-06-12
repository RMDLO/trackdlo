#!/usr/bin/env python3

import rospy
# import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg

import numpy as np
import pyrealsense2 as rs

import time
import tf2_ros
import geometry_msgs.msg

if __name__=='__main__':
    rospy.init_node ('camera_tf')

    # tf
    # base_link -> camera_color_optical_frame

    broadcaster_5 = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped_5 = geometry_msgs.msg.TransformStamped()

    static_transformStamped_5.header.stamp = rospy.Time.now()
    static_transformStamped_5.header.frame_id = "base_link"
    static_transformStamped_5.child_frame_id = "camera_color_optical_frame"

    # current camera pos
    trans_5 = [0.5308947503950723, 0.030109485611943067, 0.50]
    quat_5 = [-0.7065771296991245, 0.7075322875283535, 0.0004147019946593735, 0.012109909714664245]

    static_transformStamped_5.transform.translation.x = trans_5[0]
    static_transformStamped_5.transform.translation.y = trans_5[1]
    static_transformStamped_5.transform.translation.z = trans_5[2]

    static_transformStamped_5.transform.rotation.x = quat_5[0]
    static_transformStamped_5.transform.rotation.y = quat_5[1]
    static_transformStamped_5.transform.rotation.z = quat_5[2]
    static_transformStamped_5.transform.rotation.w = quat_5[3]

    # camera_color_optical_frame -> camera_color_frame

    broadcaster_1 = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped_1 = geometry_msgs.msg.TransformStamped()

    static_transformStamped_1.header.stamp = rospy.Time.now()
    static_transformStamped_1.header.frame_id = "camera_color_optical_frame"
    static_transformStamped_1.child_frame_id = "camera_color_frame"

    quat_1 = [0.5, -0.5, 0.5, 0.5]

    static_transformStamped_1.transform.rotation.x = quat_1[0]
    static_transformStamped_1.transform.rotation.y = quat_1[1]
    static_transformStamped_1.transform.rotation.z = quat_1[2]
    static_transformStamped_1.transform.rotation.w = quat_1[3]


    # camera_color_frame -> camera_link

    broadcaster_2 = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped_2 = geometry_msgs.msg.TransformStamped()

    static_transformStamped_2.header.stamp = rospy.Time.now()
    static_transformStamped_2.header.frame_id = "camera_color_frame"
    static_transformStamped_2.child_frame_id = "camera_link"

    trans_2 = [-0.000351057737134, -0.0148385819048, -0.000117231989861]
    quat_2 = [0.00429561594501, 0.000667857821099, -0.00226634810679, 0.999987959862]

    static_transformStamped_2.transform.translation.x = trans_2[0]
    static_transformStamped_2.transform.translation.y = trans_2[1]
    static_transformStamped_2.transform.translation.z = trans_2[2]

    static_transformStamped_2.transform.rotation.x = quat_2[0]
    static_transformStamped_2.transform.rotation.y = quat_2[1]
    static_transformStamped_2.transform.rotation.z = quat_2[2]
    static_transformStamped_2.transform.rotation.w = quat_2[3]


    # camera_link -> camera_depth_frame

    broadcaster_3 = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped_3 = geometry_msgs.msg.TransformStamped()

    static_transformStamped_3.header.stamp = rospy.Time.now()
    static_transformStamped_3.header.frame_id = "camera_link"
    static_transformStamped_3.child_frame_id = "camera_depth_frame"

    quat_3 = [0, 0, 0, 1.0]

    static_transformStamped_3.transform.rotation.x = quat_3[0]
    static_transformStamped_3.transform.rotation.y = quat_3[1]
    static_transformStamped_3.transform.rotation.z = quat_3[2]
    static_transformStamped_3.transform.rotation.w = quat_3[3]


    # camera_depth_frame -> camera_depth_optical_frame

    broadcaster_4 = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped_4 = geometry_msgs.msg.TransformStamped()

    static_transformStamped_4.header.stamp = rospy.Time.now()
    static_transformStamped_4.header.frame_id = "camera_depth_frame"
    static_transformStamped_4.child_frame_id = "camera_depth_optical_frame"

    quat_4 = [-0.5, 0.5, -0.5, 0.5]

    static_transformStamped_4.transform.rotation.x = quat_4[0]
    static_transformStamped_4.transform.rotation.y = quat_4[1]
    static_transformStamped_4.transform.rotation.z = quat_4[2]
    static_transformStamped_4.transform.rotation.w = quat_4[3]

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():

        static_transformStamped_5.header.stamp = rospy.Time.now()
        broadcaster_5.sendTransform(static_transformStamped_5)

        static_transformStamped_1.header.stamp = rospy.Time.now()
        broadcaster_1.sendTransform(static_transformStamped_1)

        static_transformStamped_2.header.stamp = rospy.Time.now()
        broadcaster_2.sendTransform(static_transformStamped_2)

        static_transformStamped_3.header.stamp = rospy.Time.now()
        broadcaster_3.sendTransform(static_transformStamped_3)

        static_transformStamped_4.header.stamp = rospy.Time.now()
        broadcaster_4.sendTransform(static_transformStamped_4)

        rate.sleep()

    # rospy.spin()