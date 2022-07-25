#!/usr/bin/env python

import time
# import rospy
import cv2
import numpy as np
import pyrealsense2 as rs
# import tf
import time
from scipy.spatial.transform import Rotation as R
import csv

# rospy.init_node('data_collection', anonymous=True)

# i = 10

# # collect image
# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()

# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = config.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()

# config.enable_stream(rs.stream.color, format = rs.format.bgr8, framerate = 30, width = 1920, height = 1080)

# # Start streaming
# pipeline.start(config)

# time.sleep(5)

# frames = pipeline.wait_for_frames()
# color_frame = frames.get_color_frame()
# color_image = np.asanyarray(color_frame.get_data())

# # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("/home/jingyixiang/test/charuco_image.jpg", color_image)
# print(np.shape(color_image))

# pipeline.stop()

# listener = tf.TransformListener()

# time.sleep(1)

# # collect tf information
# trans, quat = listener.lookupTransform('base_link', 'tool0', rospy.Time())
# print(trans)
# print(quat)

image = cv2.imread("/home/jingyixiang/test/calib_images/current/000_image.jpg")
ret, corners = cv2.findCirclesGrid(image, (4, 11), None, flags=(cv2.CALIB_CB_ASYMMETRIC_GRID))
print(ret)
cv2.drawChessboardCorners(image, (4, 11), corners, ret)

cv2.imshow('img', image)
cv2.waitKey(0) 
cv2.destroyAllWindows()
cv2.imwrite("/home/jingyixiang/test/test.jpg", image)

# raw_input("Press Enter to continue...")
# print("input received")

# def tf2csv(trans, quat, data_folder, sample_id):
# 	rot_matrix = R.from_quat(quat).as_dcm()

# 	ret = []
# 	for i in range(0, 3):
# 		for j in range(0, 3):
# 			ret.append(rot_matrix[i, j])
# 		ret.append(trans[i])

# 	ret = ret + [0.0, 0.0, 0.0, 1.0]
	
# 	# for i in range (0, 4):
# 	# 	print(*(ret[i*4 : (i+1)*4]), sep=" ")

# 	ret_str = []
# 	for i in range (0, 4):
# 		string = ''
# 		sub_arr = []
# 		for j in range (0, 4):
# 			string += str(ret[i*4 + j])
# 			if j != 3:
# 				string += " "
# 		sub_arr.append(string)
# 		ret_str.append(sub_arr)

# 	f = open(data_folder + sample_id + ".csv", 'w')
# 	writer = csv.writer(f)
# 	writer.writerows(ret_str)
# 	f.close()

# tf2csv([0.273, 0.000, 0.463], [0.706, -0.706, -0.042, 0.042], "/home/jingyixiang/test/", "test")



exit()

#==================================================================================================

start, end = 10, 20
i = start
data_folder = "/home/jingyixiang/test/calib_images/current/"

rospy.init_node('data_collection', anonymous=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.color, format = rs.format.bgr8, framerate = 30, width = 1280, height = 720)

# Start streaming
pipeline.start(config)

# initialize tf listener
listener = tf.TransformListener()
# give time for auto exposure and tf buffering
time.sleep(5)

while i < end:

	# set sample prefix
	sample_prefix = ''
	if len(str(i)) == 1:
		sample_prefix = '00'
	elif len(str(i)) == 2:
		sample_prefix =='0'
	else:
		sample_prefix == ''

	sample_id = sample_prefix + str(i)

	# wait for user input
	raw_input("--- Press Enter to start collecting data. sample_id = " + sample_id + " ---")

	# collect image
	frames = pipeline.wait_for_frames()
	color_frame = frames.get_color_frame()
	color_image = np.asanyarray(color_frame.get_data())

	# find pattern to make sure this sample is valid
	ret, corners = cv2.findCirclesGrid(color_image, (4, 11), None, flags=(cv2.CALIB_CB_ASYMMETRIC_GRID))

	if ret == False:
		print("Failed to find circle, sample invalid!")
		cv2.imshow(sample_id, image)
		cv2.waitKey(0) 
		cv2.destroyAllWindows()
	else:
		cv2.imwrite(data_folder + sample_id + "_image.jpg", image)
		print("Found pattern, sample saved!")
		cv2.drawChessboardCorners(image, (4, 11), corners, ret)
		cv2.imshow(sample_id, image)
		cv2.waitKey(0) 
		cv2.destroyAllWindows()

		# save tf data
		trans, quat = listener.lookupTransform('base_link', 'tool0', rospy.Time())
		tf2csv(trans, quat, data_folder, sample_id)

		# update id
		i += 1


