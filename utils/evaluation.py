#!/usr/bin/env python3

import numpy as np
import cv2

import rosbag
import rospy
from pylab import *
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from ros_numpy import numpify
from sensor_msgs.msg import PointCloud2, Image
import message_filters

class TrackDLOEvaluator:
    """Extracts node topic created by the TrackDLO algorithm"""
    def __init__(self):
        # self.bag = rosbag.Bag(bag)
        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.pc_sub = message_filters.Subscriber('/camera/depth/color/points', PointCloud2)
        self.trackdlo_results_sub = message_filters.Subscriber('/results_numpy', numpy_msg(Floats))
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.pc_sub], 10)
        # self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.pc_sub, self.trackdlo_results_sub], 10)
        self.ts.registerCallback(self.callback)
        self.gt_pub = rospy.Publisher('/gt_pts', numpy_msg(Floats), queue_size=10)

    def callback(self, rgb_img, pc):
    # def callback(self, rgb_img, pc, results):
        Y_true = self.get_ground_truth_nodes(rgb_img, pc)
        self.gt_pub.publish(Y_true)
        # print(results)
        # print(Y_hat)
        
    def get_ground_truth_nodes(self, rgb_img, pc):
        
        proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                                [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                                [             0.0,              0.0,               1.0, 0.0]])
        rgb_img = numpify(rgb_img)
        pc = numpify(pc)
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        
        # --- rope blue ---
        lower = (90, 60, 40)
        upper = (130, 255, 255)
        mask_blue = cv2.inRange(hsv_img, lower, upper).astype('uint8')

        # --- tape red ---
        lower = (130, 60, 40)
        upper = (180, 255, 255)
        mask_red_1 = cv2.inRange(hsv_img, lower, upper).astype('uint8')
        lower = (0, 60, 40)
        upper = (30, 255, 255)
        mask_red_2 = cv2.inRange(hsv_img, lower, upper).astype('uint8')
        mask_red = cv2.bitwise_or(mask_red_1.copy(), mask_red_2.copy()).astype('uint8')

        # Blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False
        params.filterByArea = True
        params.filterByCircularity = False
        params.filterByInertia = True
        params.filterByConvexity = False
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints_blue = detector.detect(mask_blue)
        keypoints_red = detector.detect(mask_red)
        keypoints = keypoints_blue + keypoints_red

        # Get nodes
        num_blobs = len(keypoints)
        Y_true = np.empty((num_blobs,3)) # Clarfiy variables/notation
        for i in range(num_blobs):
            x = int(keypoints[i].pt[0])
            y = int(keypoints[i].pt[1])
            pt = pc[y, x] # index pointcloud at blob pixel points
            Y_true[i, 0] = pt[0]
            Y_true[i, 1] = pt[1]
            Y_true[i, 2] = pt[2]
            
        return Y_true

    def get_piecewise_curve(Y):
        pass

    def get_piecewise_error():
        pass
    
if __name__=='__main__':
    rospy.init_node('evaluator')
    e = TrackDLOEvaluator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

# bag = '../data/rope_with_marker_folding.bag'
# data = TrackDLONodes(bag)
# nodes = data.get_data()
# print(nodes)

# ROOT_DIR = "."
# img = cv2.imread(os.path.join(ROOT_DIR,"data/images/frame0000.jpg"))

    # def get_data(self):
    #     topic_data = "/trackdlo/results_numpy"
    #     # topic_time = "/trackdlo/results"
    #     msgs_data = self.bag.read_messages(topic_data)
    #     # msgs_time = self.bag.read_messages(topic_time)
    #     time = []
    #     nodes = []
    #     # for msg_data, msg_time in zip(msgs_data, msgs_time):
    #     for msg_data in msgs_data:
    #         nodes.append(msg_data[1].data)
    #         # time.append(msg_time[1].header.stamp.secs)
    #     # convert to numpy array
    #     self.nodes = np.asarray(nodes)
    #     # self.time = np.asarray(time)
    #     return self.nodes