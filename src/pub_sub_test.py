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

if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)

    float_pub = rospy.Publisher('/float', std_msgs.msg.Float32, queue_size=10)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        float_pub.publish(6.18)
        rate.sleep()