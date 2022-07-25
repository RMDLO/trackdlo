#!/usr/bin/env python

from sensor_msgs.msg import JointState
import rospy
import pickle as pkl
import time
import os

pkl_arr = []
i = 0
cur_joint_states = []

def update_pose(data):
	global cur_joint_states
	cur_joint_states = list(data.position)

if __name__ == '__main__':
	rospy.init_node('listener', anonymous=True)

	i = 0

	main_dir = "/home/jingyixiang/test/recorded_depth/6_30_22_stick/"
	pkl_dir = main_dir + "params/"
	pkl_name = "joint_angles.json"

	try:
		os.listdir(pkl_dir)
	except:
		print("Invalid pkl directory!")
		rospy.signal_shutdown('')

	while not rospy.is_shutdown():

		# print(cur_joint_states)
		print("======================================================================")
		print("Press enter to collect and save joint angle data")
		print("Press q + enter to write to pkl file and exit program")
		key_pressed = raw_input("sample_id = " + str(i) + "\n")

		rospy.Subscriber("/joint_states", JointState, update_pose)
		time.sleep(0.2)
		
		if key_pressed == 'q':
			print('saving pickle file... \n')
			f = open(pkl_dir + pkl_name, "wb")
			pkl.dump(pkl_arr, f)
			f.close()

			print('saved with the following data: \n')
			f = open(pkl_dir + pkl_name, "rb")
			pkl_list = pkl.load(f)
			f.close()
			
			for entry in pkl_list:
				print(entry)

			rospy.signal_shutdown('')
		else:
			print("current joint angle = \n") 
			print(cur_joint_states)
			print(" ")
			pkl_arr.append(cur_joint_states)
			print("pkl arr lenth = " + str(len(pkl_arr)) + " \n")
			i += 1