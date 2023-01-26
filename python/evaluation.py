#!/usr/bin/env python3
"""
This is a Robot Operating System (ROS) node for computing piecewise error between
a set of ground truth points describing the shape of a deformable linear object
and the points predicted from a trackig algorithm node.
"""

# Python imports
import numpy as np
import cv2
from vedo import Line, Points, Arrow, Plotter

# ROS imports
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
        error, closest_pts, _ = self.get_piecewise_error(Y_track, Y_true)
        self.viz_piecewise_error(Y_true, Y_track, closest_pts)
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

    def minDistance(self, A, B, E):
        """
        Returns:
        distance: the minimum distance between point E and a line segment AB
        closest_pt_on_AB_to_E: the closest point to E on the line segment AB
        """
        # Define vectors
        AB = B - A
        BE = E - B
        AE = E - A

        # Calculate the dot product
        AB_BE = np.dot(AB, BE)
        AB_AE = np.dot(AB, AE)

        # Minimum distance from
        # point E to the line segment
        distance = 0

        # Case 1:
        # The nearest point from E on AB is point B if np.dot(AB,BE)>0
        if AB_BE > 0:
            distance = np.linalg.norm(E - B)
            closest_pt_on_AB_to_E = B
            closest_vector_to_E = E - B

        # Case 2:
        # The nearest point from E on AB is point A if np.dot(AB,AE)<0
        elif AB_AE < 0:
            distance = np.linalg.norm(E - A)
            closest_pt_on_AB_to_E = A
            closest_vector_to_E = E - A

        # Case 3:
        # If np.dot(AB,BE) or np.dot(AB,AE) = 0, then E is perpendicular
        # to the segment AB and the the perpendicular distance to E from
        # segment AB is the shortest distance.
        else:
            # Find the perpendicular distance
            intermediate_vec = np.cross(AB, AE)
            closest_vector_to_E = np.cross(AB, intermediate_vec)
            closest_pt_on_AB_to_E = E - closest_vector_to_E
            distance = np.linalg.norm(closest_vector_to_E)

        return distance, closest_pt_on_AB_to_E, closest_vector_to_E

    def get_piecewise_error(self, Y_track, Y_true):
        """
        Compute piecewise error between a set of tracked points and a set of
        ground truth points
        """
        # Should probably replace this while loop with something more robust.
        # In this bag file, the shapes are 35x3 and 37x3, respectively.
        # Put for loops into matrix form if possible.
        # For each point in Y_track, compute the distance to Y_true
        shortest_distances_to_curve = []
        closest_pts_on_Y_true = []
        closest_vectors_to_Y_true = []
        for Y in Y_track:
            distances_all_line_segments = []
            closest_pts = []
            closest_vectors = []
            for i in range(len(Y_true[:-1, 0])):
                distance, closest_pt, closest_vector = self.minDistance(
                    Y_true[i, :], Y_true[i + 1, :], Y
                )
                distances_all_line_segments.append(distance)
                closest_pts.append(closest_pt)
                closest_vectors.append(closest_vector)
            shortest_distance_to_curve_idx = np.argmin(distances_all_line_segments)
            shortest_distances_to_curve.append(
                distances_all_line_segments[shortest_distance_to_curve_idx]
            )
            closest_pts_on_Y_true.append(closest_pts[shortest_distance_to_curve_idx])
            closest_vectors_to_Y_true.append(
                closest_vectors[shortest_distance_to_curve_idx]
            )
        closest_pts_on_curve = np.asarray(closest_pts_on_Y_true)
        closest_vectors_on_curve = np.asarray(closest_vectors_to_Y_true)
        error = np.sum(shortest_distances_to_curve)

        return error, closest_pts_on_curve, closest_vectors_on_curve

    def viz_piecewise_error(self, Y_true, Y_track, closest_pts):
        Y_true_pc = Points(Y_true, c=(255, 0, 0), r=15)  # red
        Y_track_pc = Points(Y_track, c=(255, 255, 0), r=15)  # yellow
        closest_pts_pc = Points(closest_pts, c=(0, 0, 255), r=15)  # blue
        # Y_true_line = Line(Y_true_pc, c=(255, 0, 0), lw=15)
        # Y_track_line = Line(Y_track_pc, c=(255,255,0), lw=15)

        velocity_field = []

        for i in range(len(closest_pts)):
            arrow = Arrow(
                start_pt=closest_pts[i, :], end_pt=Y_track[i, :], c=(255, 255, 255)
            )
            velocity_field.append(arrow)

        plt = Plotter(N=2)
        plt.show(Y_true_pc, Y_track_pc, closest_pts_pc, velocity_field)
        plt.interactive().close()

if __name__ == "__main__":
    rospy.init_node("evaluator")
    e = TrackDLOEvaluator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
