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

cur_pcl = []
cur_joint_states = []
cur_image_arr = []

# this gives pcl in the camera's frame
def update_cur_pcl(data):
    global cur_pcl
    pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_array(data)
    cur_pcl = ros_numpy.point_cloud2.get_xyz_points(pcl_arr)

def update_pose(data):
    global cur_joint_states
    cur_joint_states = list(data.position)

def update_img(data):
    global cur_image_arr
    cur_image = ros_numpy.numpify(data)
    cur_image_arr = cv2.cvtColor(cur_image, cv2.COLOR_BGR2RGB)

# helpfer function, write to csv file
def tf2csv(trans, quat, data_folder, sample_id, sample_type):
    rot_matrix = R.from_quat(quat).as_dcm()

    ret = []
    for i in range(0, 3):
        for j in range(0, 3):
            ret.append(rot_matrix[i, j])
        ret.append(trans[i])

    ret = ret + [0.0, 0.0, 0.0, 1.0]

    ret_str = []
    for i in range (0, 4):
        string = ''
        sub_arr = []
        for j in range (0, 4):
            string += str(ret[i*4 + j])
            if j != 3:
                string += " "
        sub_arr.append(string)
        ret_str.append(sub_arr)

    f = open(data_folder + sample_id + "_" + sample_type + ".csv", 'w')
    writer = csv.writer(f)
    writer.writerows(ret_str)
    f.close()

def record(main_dir, start=0, save_image=False, save_joint_states=False, save_cam_pose=False, load_existing_pkl=False):
    listener = tf.TransformListener()

    i = start

    pkl_arr = []
    if save_joint_states and load_existing_pkl:
        f = open(main_dir + "params/joint_angles.json", "rb")
        pkl_arr = pkl.load(f)
        f.close()

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
            if save_joint_states:
                print('saving pickle file... \n')
                f = open(main_dir + "params/joint_angles.json", "wb")
                pkl.dump(pkl_arr, f)
                f.close()

                print('saved with the following data: \n')
                f = open(main_dir + "params/joint_angles.json", "rb")
                pkl_list = pkl.load(f)
                f.close()
                
                for entry in pkl_list:
                    print(entry)
            else:
                print("Shutting down... \n")

            rospy.signal_shutdown('')
        else:
            rospy.Subscriber("/camera/color/image_raw", Image, update_img)
            time.sleep(0.2)

            if save_image:
                if len(cur_image_arr) == 0:
                    print(" ")
                    print("Could not capture image, please try again! \n")
                    continue
                cv2.imwrite(main_dir + sample_id + "_rgb.png", cur_image_arr)

            rospy.Subscriber("/camera/depth/color/points", PointCloud2, update_cur_pcl)
            time.sleep(0.2)

            # sometimes pointcloud can be empty
            if len(cur_pcl) == 0:
                print(" ")
                print("Could not capture point cloud, please try again! \n")
                continue

            trans, quat = listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time())

            f = open(main_dir + sample_id + "_pcl.json", "wb")
            pkl.dump(cur_pcl, f)
            f.close()

            if save_joint_states:
                rospy.Subscriber("/joint_states", JointState, update_pose)
                time.sleep(0.2)
                print("current joint angle = \n") 
                print(cur_joint_states)
                print(" ")
                # to prevent data lost caused by program crash
                if len(pkl_arr) == 0:
                    pkl_arr.append(cur_joint_states)
                    print("pkl arr lenth = " + str(len(pkl_arr)))
                    f = open(main_dir + "params/joint_angles.json", "wb")
                    pkl.dump(pkl_arr, f)
                    f.close()
                else:
                    f = open(main_dir + "params/joint_angles.json", "rb")
                    pkl_arr = pkl.load(f)
                    f.close()
                    pkl_arr.append(cur_joint_states)
                    print("pkl arr lenth = " + str(len(pkl_arr)))
                    f = open(main_dir + "params/joint_angles.json", "wb")
                    pkl.dump(pkl_arr, f)
                    f.close()

            if save_cam_pose:
                tf2csv(trans, quat, main_dir, sample_id, "cam_pose")

                trans, quat = listener.lookupTransform('base_link', 'tool0', rospy.Time())
                tf2csv(trans, quat, main_dir, sample_id, "tool0_pose")

            print("Data saved successfully! \n")
            i += 1

if __name__ == '__main__':
    rospy.init_node('record_pcl', anonymous=True)
    main_dir = setting_path = join(dirname(dirname(abspath(__file__))), "data/")

    # try:
    os.listdir(main_dir)
    os.listdir(main_dir + 'params/')
    print("######################################################################")
    print("Collected data will be saved at the following directory:")
    print(main_dir)
    print("###################################################################### \n")

    record(main_dir, start=0, save_image=True, save_joint_states=False, save_cam_pose=False, load_existing_pkl=False)
    # except:
    # 	print("Invalid directory!")
    # 	rospy.signal_shutdown('')