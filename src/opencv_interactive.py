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
import pyrealsense2 as rs


rect = (0,0,0,0)
startPoint = False
endPoint = False

# this mask will get updated each iteration
mouse_mask = None

def on_mouse(event, x, y, flags, params):

    global rect, startPoint, endPoint

    # get mouse click
    if event == cv2.EVENT_LBUTTONDOWN:

        if startPoint == True and endPoint == True:
            startPoint = False
            endPoint = False
            rect = (0, 0, 0, 0)

        if startPoint == False:
            rect = (x, y, x, y)
            startPoint = True
        elif endPoint == False:
            rect = (rect[0], rect[1], x, y)
            endPoint = True
    
    # draw rectangle when mouse hovering
    elif event == cv2.EVENT_MOUSEMOVE and startPoint == True and endPoint == False:
        rect = (rect[0], rect[1], x, y)

def callback (rgb):
    global rect, startPoint, endPoint, mouse_mask

    cur_image = ros_numpy.numpify(rgb)

    # convert color for opencv display
    cur_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_BGR2RGB)

    # frame = cur_image.copy()
    # resize for smaller window
    height, width, layers = cur_image.shape
    new_h = int(height / 1.5)
    new_w = int(width / 1.5)
    frame = cv2.resize(cur_image, (new_w, new_h))

    # initialize mask if none
    if mouse_mask is None:
        mouse_mask = np.ones(frame.shape)

    # filter with mask
    frame = (frame * mouse_mask).astype('uint8')

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)    

    key = cv2.waitKey(10)

    if key == 114: # r
        # reset everyhting
        frame = cv2.resize(cur_image, (new_w, new_h))
        startPoint = False
        endPoint = False
        mouse_mask = np.ones(frame.shape)
        cv2.imshow('frame',frame)
    else:
        #drawing rectangle
        if startPoint == True and endPoint != True:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
        
        # if another rectangle is drawn, update mask
        if startPoint == True and endPoint == True:
            mouse_mask[rect[1]:rect[3], rect[0]:rect[2], :] = 0

        cv2.imshow('frame', frame)

    # publish mask
    occlusion_mask = (mouse_mask*255).astype('uint8')

    # resize back for pub
    occlusion_mask = cv2.resize(occlusion_mask, (width, height))

    occlusion_mask_img_msg = ros_numpy.msgify(Image, occlusion_mask, 'rgb8')
    occlusion_mask_img_pub.publish(occlusion_mask_img_msg)


if __name__=='__main__':
    rospy.init_node('test', anonymous=True)

    rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, callback)

    tracking_img_pub = rospy.Publisher ('/tracking_img', Image, queue_size=10)
    occlusion_mask_img_pub = rospy.Publisher('/mask_with_occlusion', Image, queue_size=10)

    rospy.spin()