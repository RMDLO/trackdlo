#!/usr/bin/env python

#####################################################################
# Description:                                                      #
# This program uses OpenCV 4.2 to perform ChArUco board pose        #
# estimation and publish the corresponding tf messages to /tf. An   #
# image with detected markers and corners drawn is also published.  #
# Not using the charuco_detector due to buggy OpenCV 3.2 (default   #
# (OpenCV version for C++) ChArUco detector.                        #
#####################################################################

import time
import rospy
import cv2
import numpy as np
import pyrealsense2 as rs
import time
from scipy.spatial.transform import Rotation as R
import tf2_ros
import geometry_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

# create ChArUco board with the specified config
def create_board(board_dict, square_len, marker_len, num_sq_x, num_sq_y):

	aruco_dict = cv2.aruco.Dictionary_get(board_dict)
	board = cv2.aruco.CharucoBoard_create(num_sq_x, num_sq_y, square_len, marker_len, aruco_dict)
	arucoParams = cv2.aruco.DetectorParameters_create()

	return aruco_dict, board, arucoParams


# perform detection and publish info
def detect(img_width, img_height, square_len, marker_len, num_sq_x, num_sq_y):

	aruco_dict, board, arucoParams = create_board(cv2.aruco.DICT_4X4_250, square_len, marker_len, num_sq_x, num_sq_y)

	# start stream
	# Configure color stream
	pipeline = rs.pipeline()
	config = rs.config()

	pipeline_wrapper = rs.pipeline_wrapper(pipeline)
	pipeline_profile = config.resolve(pipeline_wrapper)
	device = pipeline_profile.get_device()

	config.enable_stream(rs.stream.color, format = rs.format.bgr8, framerate = 30, width = img_width, height = img_height)

	# Start streaming
	pipeline.start(config)

	# get camera intrinsics
	profile = pipeline.get_active_profile()
	color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
	color_intrinsics = color_profile.get_intrinsics()
	w, h = color_intrinsics.width, color_intrinsics.height
	cx, cy = color_intrinsics.ppx, color_intrinsics.ppy
	fx, fy = color_intrinsics.fx, color_intrinsics.fy

	dist_coeffs = np.array(color_intrinsics.coeffs)
	camera_matrix = np.array([[ fx, 0.0,  cx], \
							  [0.0,  fy,  cy], \
							  [0.0, 0.0, 1.0]])

	# initialize tf broadcaster
	broadcaster = tf2_ros.TransformBroadcaster()
	transformStamped = geometry_msgs.msg.TransformStamped()

	# initialize image publisher
	detection = rospy.Publisher('charuco_detection', Image, queue_size=10)
	corner_arr_pub = rospy.Publisher('corner_uv', String, queue_size=10)

	rate = rospy.Rate(100)

	# keep running detection
	while not rospy.is_shutdown():

		# collect image from stream
		frames = pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()
		color_image = np.asanyarray(color_frame.get_data())

		# print(np.shape(color_image))

		# ChArUco detection
		gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
		corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)  
		cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)

		# image message
		img_msg = Image()
		img_msg.header.stamp = rospy.Time.now()
		img_msg.height = img_height
		img_msg.width = img_width
		img_msg.encoding = 'bgr8'
		img_msg.is_bigendian = False
		img_msg.step = 3 * img_width

		corner_arr = ''

		# if there is at least one marker detected
		if ids is not None: 
			charucoretval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
			im_with_charuco_board = cv2.aruco.drawDetectedCornersCharuco(color_image, charucoCorners, charucoIds)
			im_with_charuco_board = cv2.aruco.drawDetectedMarkers(im_with_charuco_board, corners, ids)

			# print(np.shape(charucoCorners))
			try:
				corner_arr = str(charucoCorners[1, 0, 0]) + ' ' + str(charucoCorners[1, 0, 1])
				corner_arr_pub.publish(corner_arr)
			except:
				print('cannot find corners!')

			# ChArUco board pose estimation
			retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, dist_coeffs, None, None)

			if retval == True:
				im_with_charuco_board = cv2.aruco.drawAxis(im_with_charuco_board, camera_matrix, dist_coeffs, rvec, tvec, 5)  
				rot, _ = cv2.Rodrigues(rvec)
				r = R.from_dcm(rot)
				rquat = r.as_quat()

				# publish tf message
				transformStamped.header.stamp = rospy.Time.now()
				transformStamped.header.frame_id = "camera_color_optical_frame"
				transformStamped.child_frame_id = "charuco"

				transformStamped.transform.translation.x = tvec[0]/100.0
				transformStamped.transform.translation.y = tvec[1]/100.0
				transformStamped.transform.translation.z = tvec[2]/100.0

				transformStamped.transform.rotation.x = rquat[0]
				transformStamped.transform.rotation.y = rquat[1]
				transformStamped.transform.rotation.z = rquat[2]
				transformStamped.transform.rotation.w = rquat[3]

				broadcaster.sendTransform(transformStamped)

				# publish img message
				img_msg.data = im_with_charuco_board.tobytes()
				detection.publish(img_msg)

		else:
			img_msg.data = color_image.tobytes()
			detection.publish(img_msg)

		rate.sleep()


if __name__ == '__main__':
	rospy.init_node('python_charuco_detector', anonymous=True)
	# detect(1920, 1080, 2, 1.5, 10, 14)
	detect(1920, 1080, 4, 3, 5, 7)
	rospy.spin()

	
