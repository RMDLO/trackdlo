#!/usr/bin/env python

import time
import rospy
import cv2
import numpy as np
import pyrealsense2 as rs
import tf
import time
from scipy.spatial.transform import Rotation as R

rospy.init_node('print_tf', anonymous=True)

listener = tf.TransformListener()
time.sleep(1)

trans, quat = listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time())
r = R.from_quat(quat)

print(trans)
print(r.as_euler('zyx', degrees=True))
print(quat)