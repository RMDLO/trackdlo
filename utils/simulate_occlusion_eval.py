#!/usr/bin/env python3

import rospy
import ros_numpy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
import cv2
import numpy as np

def callback (arr_msg):
    arr = arr_msg.data

    mouse_mask = np.ones((720, 1280, 3))
    mouse_mask[arr[1]:arr[3], arr[0]:arr[2], :] = 0

    # publish mask
    occlusion_mask = (mouse_mask*255).astype('uint8')

    occlusion_mask_img_msg = ros_numpy.msgify(Image, occlusion_mask, 'rgb8')
    occlusion_mask_img_pub.publish(occlusion_mask_img_msg)


if __name__=='__main__':
    rospy.init_node('simulate_occlusion_eval', anonymous=True)

    arr_sub = rospy.Subscriber('/corners', Int32MultiArray, callback)
    occlusion_mask_img_pub = rospy.Publisher('/mask_with_occlusion', Image, queue_size=10)

    rospy.spin()