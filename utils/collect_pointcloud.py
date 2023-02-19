#!/usr/bin/env python3

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
from os.path import dirname, abspath, join

cur_pc = []
cur_image_arr = []
cur_result = []
cur_tracking_image_arr = []

# this gives pcl in the camera's frame
def update_cur_pc(data):
    global cur_pc
    pc_arr = ros_numpy.point_cloud2.pointcloud2_to_array(data)
    cur_pc = ros_numpy.point_cloud2.get_xyz_points(pc_arr)

def update_cur_result(data):
    global cur_result
    result_arr = ros_numpy.point_cloud2.pointcloud2_to_array(data)
    cur_result = ros_numpy.point_cloud2.get_xyz_points(result_arr)

def update_img(data):
    global cur_image_arr
    cur_image = ros_numpy.numpify(data)
    cur_image_arr = cv2.cvtColor(cur_image, cv2.COLOR_BGR2RGB)

def update_tracking_img(data):
    global cur_tracking_image_arr
    cur_tracking_image_arr = ros_numpy.numpify(data)
    # cur_tracking_image_arr = cv2.cvtColor(cur_tracking_image, cv2.COLOR_BGR2RGB)

def record(main_dir, start=0, save_image=False, save_results=False):

    i = start

    while not rospy.is_shutdown():
        sample_id = ''
        if len(str(i)) == 1:
            sample_id = '00' + str(i)
        else:
            sample_id = '0' + str(i)

        print("======================================================================")
        print("Press enter to collect and save point cloud and camera pose data")
        print("Press q + enter to exit the program")
        key_pressed = input("sample_id = " + sample_id + "\n")

        if key_pressed == 'q':
            print("Shutting down... \n")
            rospy.signal_shutdown('')
        else:
            rospy.Subscriber("/camera/color/image_raw", Image, update_img)
            rospy.Subscriber("/tracking_img", Image, update_tracking_img)
            rospy.Subscriber("/trackdlo_results_pc", PointCloud2, update_cur_result)
            rospy.Subscriber("/camera/depth/color/points", PointCloud2, update_cur_pc)

            if save_image:
                if len(cur_image_arr) == 0:
                    print(" ")
                    print("Could not capture image, please try again! \n")
                    continue
                cv2.imwrite(main_dir + sample_id + "_rgb.png", cur_image_arr)
            
            if save_results:
                if len(cur_result) == 0:
                    print(" ")
                    print("Could not capture results, please try again! \n")
                    continue

                f = open(main_dir + sample_id + "_results.json", "wb")
                pkl.dump(cur_result, f)
                f.close()

                if len(cur_tracking_image_arr) == 0:
                    print(" ")
                    print("Could not capture tracking image, please try again! \n")
                    continue
                cv2.imwrite(main_dir + sample_id + "_result.png", cur_tracking_image_arr)

            # sometimes pointcloud can be empty
            if len(cur_pc) == 0:
                print(" ")
                print("Could not capture point cloud, please try again! \n")
                continue

            f = open(main_dir + sample_id + "_pc.json", "wb")
            pkl.dump(cur_pc, f)
            f.close()

            print("Data saved successfully! \n")
            i += 1

if __name__ == '__main__':
    rospy.init_node('record_data', anonymous=True)
    main_dir = setting_path = join(dirname(dirname(abspath(__file__))), "data/")

    # try:
    os.listdir(main_dir)
    print("######################################################################")
    print("Collected data will be saved at the following directory:")
    print(main_dir)
    print("###################################################################### \n")

    record(main_dir, start=0, save_image=True, save_results=True)
    # except:
    # 	print("Invalid directory!")
    # 	rospy.signal_shutdown('')