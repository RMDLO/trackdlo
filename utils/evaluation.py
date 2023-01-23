import numpy as np
import cv2

import rosbag
import rospy
from pylab import *
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

class TrackDLONodes(object):
    """Extracts node topic created by the TrackDLO algorithm"""
    def __init__(self, bag):
        self.bag = rosbag.Bag(bag)
        self.numpy_pub = rospy.Publisher('/results_numpy', numpy_msg(Floats), queue_size=10)
    def get_data(self):
        topic_data = "/trackdlo/results_numpy"
        # topic_time = "/trackdlo/results"
        msgs_data = self.bag.read_messages(topic_data)
        # msgs_time = self.bag.read_messages(topic_time)
        time = []
        nodes = []
        # for msg_data, msg_time in zip(msgs_data, msgs_time):
        for msg_data in msgs_data:
            nodes.append(msg_data[1].data)
            # time.append(msg_time[1].header.stamp.secs)
        # convert to numpy array
        self.nodes = np.asarray(nodes)
        # self.time = np.asarray(time)
        return self.nodes


def get_ground_truth_nodes(hsv_img, pc):
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
    Y_true = np.array() # Clarfiy variables/notation
    for i in range(num_blobs):
        x = int(keypoints[i].pt[0])
        y = int(keypoints[i].pt[1])
        z = # index pointcloud
        Y_true[i, 0] = x
        Y_true[i, 1] = y
        Y_true[i, 2] = z

    return Y_true


def get_piecewise_curve(Y):
    pass

def get_piecewise_error():
    pass

bag = '../data/rope_with_marker_folding.bag'
data = TrackDLONodes(bag)
nodes = data.get_data()
print(nodes)

img = cv2.imread("../data/images/rope_with_marker_stationary_curved_images/frame0000.jpg") # Make path more robust
hsv_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2HSV)
pc = None
Y_true = get_ground_truth_nodes(hsv_img, pc)
