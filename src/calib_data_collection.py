#!/usr/bin/env python

import time
import rospy
import cv2
import numpy as np
import pyrealsense2 as rs
import tf
import time
from scipy.spatial.transform import Rotation as R
import csv
import pickle as pkl
from sensor_msgs.msg import JointState

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


cur_joint_states = []

def update_pose(data):
	global cur_joint_states
	cur_joint_states = list(data.position)

def collect(data_folder, pattern, rows, cols, w, h, start=0, save_joint_states=False, load_existing_pkl=False):

	i = start

	pkl_arr = []
	if save_joint_states and load_existing_pkl:
		f = open(data_folder + "params/joint_angles.json", "rb")
		pkl_arr = pkl.load(f)
		f.close()

	rospy.init_node('data_collection', anonymous=True)

	# Configure color stream
	pipeline = rs.pipeline()
	config = rs.config()

	pipeline_wrapper = rs.pipeline_wrapper(pipeline)
	pipeline_profile = config.resolve(pipeline_wrapper)
	device = pipeline_profile.get_device()

	config.enable_stream(rs.stream.color, format = rs.format.bgr8, framerate = 30, width = w, height = h)

	# Start streaming
	pipeline.start(config)

	# initialize tf listener
	listener = tf.TransformListener()
	# give time for auto exposure and tf buffering
	time.sleep(5)

	while not rospy.is_shutdown():

		# set sample prefix
		sample_prefix = ''
		if len(str(i)) == 1:
			sample_prefix = '00'
		elif len(str(i)) == 2:
			sample_prefix ='0'
		else:
			sample_prefix = ''

		sample_id = sample_prefix + str(i)

		# wait for user input
		print("======================================================================")
		print("Press enter to start collecting data")
		print("Press q + enter to exit program")
		key_pressed = raw_input("sample_id = " + sample_id + " \n")
		if key_pressed == 'q':

			if save_joint_states:
				print('saving pickle file... \n')
				f = open(data_dir + "params/joint_angles.json", "wb")
				pkl.dump(pkl_arr, f)
				f.close()

				print('saved with the following data: \n')
				f = open(data_dir + "params/joint_angles.json", "rb")
				pkl_list = pkl.load(f)
				f.close()
				
				for entry in pkl_list:
					print(entry)
			else:
				print("Shutting down... \n")

			rospy.signal_shutdown('')

		else:
			# collect image
			frames = pipeline.wait_for_frames()
			color_frame = frames.get_color_frame()
			color_image = np.asanyarray(color_frame.get_data())
			im = color_image.copy()

			ret = False

			if pattern == 'circle':
				# find pattern to make sure this sample is valid
				ret, corners = cv2.findCirclesGrid(color_image, (rows, cols), None, flags=(cv2.CALIB_CB_ASYMMETRIC_GRID))
			elif pattern == 'chessboard':
				ret, corners = cv2.findChessboardCorners(color_image, (rows, cols), None)

			if ret == False:
				print("Failed to find pattern, sample invalid!")
				small_img = cv2.resize(color_image, (int(w/2.0), int(h/2.0)))
				cv2.imshow(sample_id, small_img)
				cv2.waitKey(0) 
				cv2.destroyAllWindows()
			else:
				cv2.drawChessboardCorners(color_image, (rows, cols), corners, ret)
				small_img = cv2.resize(color_image, (int(w/2.0), int(h/2.0)))
				cv2.imshow(sample_id, small_img)
				cv2.waitKey(0) 
				cv2.destroyAllWindows()

				print("Found pattern, would you like to save this sample?")
				key_pressed = raw_input("Press s + enter to save sample, press enter to retake sample. \n")

				if key_pressed == 's':

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
							f = open(data_dir + "params/joint_angles.json", "wb")
							pkl.dump(pkl_arr, f)
							f.close()
						else:
							f = open(data_dir + "params/joint_angles.json", "rb")
							pkl_arr = pkl.load(f)
							f.close()
							pkl_arr.append(cur_joint_states)
							print("pkl arr lenth = " + str(len(pkl_arr)))
							f = open(data_dir + "params/joint_angles.json", "wb")
							pkl.dump(pkl_arr, f)
							f.close()

					# save image
					cv2.imwrite(data_folder + sample_id + "_image.jpg", im)
					# save tf data for hand pose
					trans, quat = listener.lookupTransform('base_link', 'tool0', rospy.Time())
					tf2csv(trans, quat, data_folder, sample_id, "pose")

					# update id
					i += 1
				else:
					print("Sample rejected")

		print(" ")

	pipeline.stop()

if __name__ == '__main__':
	data_folder = "/home/jingyixiang/test/calib_images/6_27_22_1920x1080/"
	collect(data_folder, pattern='chessboard', rows=4, cols=6, w=1280, h=720, start=0, save_joint_states=False, load_existing_pkl=False)