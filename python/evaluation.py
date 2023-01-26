#!/usr/bin/env python3
"""
This is a Robot Operating System (ROS) node for computing piecewise error between
a set of ground truth points describing the shape of a deformable linear object
and the points predicted from a trackig algorithm node.
"""

# Python imports
import numpy as np
import cv2
from math import sqrt
import matplotlib.pyplot as plt

# ROS imports
import rosbag
import rospy
from ros_numpy import point_cloud2
from ros_numpy import numpify
from sensor_msgs.msg import PointCloud2, Image
import message_filters

# TrackDLO imports
from trackdlo import sort_pts

class TrackDLOEvaluator:
    """
    Extracts predicted nodes published by the tracking algorithm,
    obtains ground truth node positions through color thresholding and exttracting blob
    centroids, and computes piecewise error.
    """

    def __init__(self):
        # self.bag = rosbag.Bag(bag)
        self.rgb_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.pc_sub = message_filters.Subscriber(
            "/camera/depth/color/points", PointCloud2
        )
        self.trackdlo_results_sub = message_filters.Subscriber(
            "/results_pc", PointCloud2
        )
        self.ts = message_filters.TimeSynchronizer(
            [self.rgb_sub, self.pc_sub, self.trackdlo_results_sub], 10
        )
        self.ts.registerCallback(self.callback)
        self.gt_pub = rospy.Publisher("/gt_pts", PointCloud2, queue_size=10)

    def callback(self, rgb_img, pc, track):
        """
        Callback function which processes raw RGB Image data, raw point cloud data, and
        tracked point cloud data for evaluation.
        """
        Y_true = self.get_ground_truth_nodes(rgb_img, pc)
        Y_track = self.get_tracking_nodes(track)
        slopes, intercepts = self.get_piecewise_curve(Y_true)
        error = self.get_piecewise_error(Y_track, Y_true, slopes, intercepts)
        print(error)

    def get_ground_truth_nodes(self, rgb_img, pc):
        """
        Compute the ground truth node positions on rope with colored marker tape with
        color thresholding and extracting blob centroids and publish the nodes as a
        PointCloud2 message.
        """
        head = rgb_img.header

        rgb_img = numpify(rgb_img)
        pc = numpify(pc)
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

        # --- rope blue ---
        lower = (90, 60, 40)
        upper = (130, 255, 255)
        mask_blue = cv2.inRange(hsv_img, lower, upper).astype("uint8")

        # --- tape red ---
        lower = (130, 60, 40)
        upper = (180, 255, 255)
        mask_red_1 = cv2.inRange(hsv_img, lower, upper).astype("uint8")
        lower = (0, 60, 40)
        upper = (30, 255, 255)
        mask_red_2 = cv2.inRange(hsv_img, lower, upper).astype("uint8")
        mask_red = cv2.bitwise_or(mask_red_1.copy(), mask_red_2.copy()).astype("uint8")

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
        Y_true = np.empty((num_blobs, 3))  # Clarfiy variables/notation
        Y_true_msg = PointCloud2()
        pc_list = []
        for i in range(num_blobs):
            x = int(keypoints[i].pt[0])
            y = int(keypoints[i].pt[1])
            # index pointcloud at blob pixel points:
            new_pt = pc[y, x].tolist()
            Y_true[i, 0] = new_pt[0]
            Y_true[i, 1] = new_pt[1]
            Y_true[i, 2] = new_pt[2]
            pc_list.append(np.array(new_pt[0:3]).astype(np.float32))
        pc = np.vstack(pc_list).astype(np.float32).T
        pc = sort_pts(pc)
        Y_true = sort_pts(Y_true)
        rec_project = np.core.records.fromarrays(
            pc, names="x, y, z", formats="float32, float32, float32"
        )

        Y_true_msg = point_cloud2.array_to_pointcloud2(
            rec_project, head.stamp, frame_id="camera_color_optical_frame"
        )
        self.gt_pub.publish(Y_true_msg)

        return Y_true

    def get_tracking_nodes(self, track):
        """
        Convert tracking output from PointCloud2 message to numpy array
        """
        pc_data = point_cloud2.pointcloud2_to_array(track)
        pc = point_cloud2.get_xyz_points(pc_data)
        return pc

    def get_piecewise_curve(self, Y_true):
        """
        Create arrays of parameters describing line segments between subsequent
        pairs of points describing the shape of a DLO
        """
        x = Y_true[:, 0]
        y = Y_true[:, 1]
        z = Y_true[:, 2]
        # 2D:
        slopes = np.divide(y[1:] - y[:-1], x[1:] - x[:-1])
        intercepts = y[1:] - np.multiply(slopes, x[1:])
        ##### 3D: IMPLEMENT ME #####
        ## in parametric form:
        # vector = Y_true[1:,:] - Y_true[:-1,:]
        # intercept = Y[:-1,:]
        # Note this is not cartesian form!
        return slopes, intercepts
 
    def minDistance(self, A, B, E) :
        '''
        Return the minimum distance between point E and a line segment defined
        by points AB
        '''
        # Define vectors
        AB = B - A
        BE = E - B
        AE = E - A
    
        # Calculate the dot product
        AB_BE = np.dot(AB,BE)
        AB_AE = np.dot(AB,AE)
    
        # Minimum distance from
        # point E to the line segment
        distance = 0
    
        # Case 1: 
        # The nearest point from E on AB is point B if np.dot(AB,BE)>0
        if (AB_BE > 0) :
            distance = np.linalg.norm(E-B)
    
        # Case 2:
        # The nearest point from E on AB is point A if np.dot(AB,AE)<0
        elif (AB_AE < 0) :
            distance = np.linalg.norm(E-A)
    
        # Case 3:
        # If np.dot(AB,BE) or np.dot(AB,AE) = 0, then E is perpendicular
        # to the segment AB and the the perpendicular distance to E from
        # segment AB is the shortest distance.
        else:
            # Find the perpendicular distance
            distance = np.linalg.norm(np.cross(AB,AE))
        
        return distance

    def get_piecewise_error(self, Y_track, Y_true, slopes, intercepts):
        """
        Compute piecewise error between a set of tracked points and a set of
        ground truth points
        """
        # Should probably replace this while loop with something more robust.
        # In this bag file, the shapes are 35x3 and 37x3, respectively.
        # Put for loops into matrix form if possible. 
        # For each point in Y_track, compute the distance to Y_true
        shortest_distances_to_curve = []
        for Y in Y_track:
            # Compute the distance from (x0,y0) to every line segment from Y_true:
            distances_all_line_segments = []
            for i in range(len(Y_true[:-1,0])):
                distance = self.minDistance(Y_true[i,:],Y_true[i+1,:],Y)
                distances_all_line_segments.append(distance)
            shortest_distance_to_curve = np.min(distances_all_line_segments)
            shortest_distances_to_curve.append(shortest_distance_to_curve)
        error = np.sum(shortest_distances_to_curve)

        return error


if __name__ == "__main__":
    rospy.init_node("evaluator")
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
