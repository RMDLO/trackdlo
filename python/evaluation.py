#!/usr/bin/env python3
"""
This is a Robot Operating System (ROS) node for computing piecewise error between
a set of ground truth points describing the shape of a deformable linear object
and the points predicted from a tracking algorithm node.
"""

# Python imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from vedo import Line, Points, Arrow, Plotter

# ROS imports
import rospy
import rosbag
from ros_numpy import point_cloud2
from ros_numpy import numpify
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Float64
import message_filters


class TrackDLOEvaluator:
    """
    Extracts predicted nodes published by the tracking algorithm,
    obtains ground truth node positions through color thresholding and extracting blob
    centroids, and computes piecewise error.
    """

    def __init__(self, rgb_length):
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
        self.gt_pub = rospy.Publisher("/gt_pts", PointCloud2, queue_size=100)
        self.error_pub = rospy.Publisher("/error", Float64, queue_size=100)
        self.cumulative_error = 0
        self.error_list = []
        self.length = rgb_length

    def callback(self, rgb_img, pc, track):
        """
        Callback function which processes raw RGB Image data, raw point cloud data, and
        tracked point cloud data for evaluation.
        """
        Y_true, head = self.get_ground_truth_nodes(rgb_img, pc)
        Y_track = self.get_tracking_nodes(track)
        # self.viz_piecewise_error(Y_true, Y_track, closest_pts)
        self.final_error(Y_true, Y_track, head)

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
        pc = self.sort_pts(pc)
        Y_true = self.sort_pts(Y_true)
        rec_project = np.core.records.fromarrays(
            pc, names="x, y, z", formats="float32, float32, float32"
        )

        Y_true_msg = point_cloud2.array_to_pointcloud2(
            rec_project, head.stamp, frame_id="camera_color_optical_frame"
        )
        self.gt_pub.publish(Y_true_msg)

        return Y_true, head

    def sort_pts(self, Y_0):
        """
        Sort points in a point cloud
        """
        diff = Y_0[:, None, :] - Y_0[None, :,  :]
        diff = np.square(diff)
        diff = np.sum(diff, 2)

        N = len(diff)
        G = diff.copy()

        selected_node = np.zeros(N,).tolist()
        selected_node[0] = True
        Y_0_sorted = []
            
        reverse = 0
        counter = 0
        reverse_on = 0
        insertion_counter = 0
        last_visited_b = 0
        while (counter < N - 1):
            
            minimum = 999999
            a = 0
            b = 0
            for m in range(N):
                if selected_node[m]:
                    for n in range(N):
                        if ((not selected_node[n]) and G[m][n]):  
                            # not in selected and there is an edge
                            if minimum > G[m][n]:
                                minimum = G[m][n]
                                a = m
                                b = n

            if len(Y_0_sorted) == 0:
                Y_0_sorted.append(Y_0[a].tolist())
                Y_0_sorted.append(Y_0[b].tolist())
            else:
                if last_visited_b != a:
                    reverse += 1
                    reverse_on = a
                    insertion_counter = 0

                if reverse % 2 == 1:
                    # switch direction
                    Y_0_sorted.insert(Y_0_sorted.index(Y_0[a].tolist()), Y_0[b].tolist())
                elif reverse != 0:
                    Y_0_sorted.insert(Y_0_sorted.index(Y_0[reverse_on].tolist())+1+insertion_counter, Y_0[b].tolist())
                    insertion_counter += 1
                else:
                    Y_0_sorted.append(Y_0[b].tolist())

            last_visited_b = b
            selected_node[b] = True

            counter += 1

        return np.array(Y_0_sorted)

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
        AE = E - A
        
        distance = np.linalg.norm(np.cross(AE, AB))/np.linalg.norm(AB)
        closest_pt_on_AB_to_E = A + AB*np.dot(AE, AB)/np.dot(AB, AB)
        
        # Check whether point is on line segment.
        # If point is not on line segment, find the nearest endpoint.
        AP = closest_pt_on_AB_to_E - A
        if np.dot(AP, AB) < 0 or np.dot(AP, AB) > np.dot(AB, AB):
            BE = E - B
            distance_AE = np.sqrt(np.dot(AE, AE))
            distance_BE = np.sqrt(np.dot(BE, BE))
            if distance_AE > distance_BE:
                distance = distance_BE
                closest_pt_on_AB_to_E = B
            else:
                distance = distance_AE
                closest_pt_on_AB_to_E = A

        return distance, closest_pt_on_AB_to_E

    def get_piecewise_error(self, Y_track, Y_true):
        """
        Compute piecewise error between a set of tracked points and a set of
        ground truth points
        """
        # Should probably replace this while loop with something more robust.
        # In this bag file, the shapes are 35x3 and 37x3, respectively.
        # Put for loops into matrix form if possible.
        # For each point in Y_track, compute the distance to Y_true.
        # Calculate weights based on line segment half-lengths.
        shortest_distances_to_curve = []
        closest_pts_on_Y_true = []
        weights = []
        for idx, Y in enumerate(Y_track):
            dist = None
            closest_pt = None
            for i in range(len(Y_true)-1):
                dist_i, closest_pt_i = self.minDistance(
                    Y_true[i, :], Y_true[i + 1, :], Y
                )
                if dist == None or dist_i < dist:
                    dist = dist_i
                    closest_pt = closest_pt_i
            weight = 0
            # For endpoints, only weight by one half-length.
            # For all other points, weight by both adjoining half-lengths.
            if idx == 0:
                weight += np.linalg.norm(Y_track[idx, :] - Y_track[idx + 1, :]) / 2
            elif idx == len(Y_track) - 1:
                weight += np.linalg.norm(Y_track[idx, :] - Y_track[idx - 1, :]) / 2
            else:
                weight += np.linalg.norm(Y_track[idx, :] - Y_track[idx + 1, :]) / 2
                weight += np.linalg.norm(Y_track[idx, :] - Y_track[idx - 1, :]) / 2
            shortest_distances_to_curve.append(dist)
            closest_pts_on_Y_true.append(closest_pt)
            weights.append(weight)
        closest_pts_on_curve = np.asarray(closest_pts_on_Y_true)
        error_frame = np.sum(np.multiply(shortest_distances_to_curve, weights))

        return error_frame

    def final_error(self, Y_track, Y_true, head):
        E1 = self.get_piecewise_error(Y_track, Y_true)
        E2 = self.get_piecewise_error(Y_true, Y_track)
        self.cumulative_error = self.cumulative_error + (E1 + E2)/2

        error_msg = Float64()
        error_msg.data = self.cumulative_error
        self.error_list.append(self.cumulative_error)
        self.error_pub.publish(error_msg)

        print(len(self.error_list), self.length)
        if len(self.error_list) >= self.length-10:
            plt.clf()
            plt.plot(self.error_list)
            plt.savefig('/home/hollydinkel/rmdlo_tracking/src/trackdlo/data/output/eval.png')

    def viz_piecewise_error(self, Y_true, Y_track, closest_pts):
        '''
        Visualize piecewise error using Vedo 3D plotting library
        '''
        Y_true_pc = Points(Y_true, c=(255, 220, 0), r=15)  # red
        Y_track_pc = Points(Y_track, c=(0, 0, 0), r=15)  # yellow
        closest_pts_pc = Points(closest_pts, c=(255, 0, 0), r=15)  # blue
        Y_true_line = Line(Y_true_pc, c=(255, 220, 0), lw=5)
        Y_track_line = Line(Y_track_pc, c=(0, 0, 0), lw=5)

        velocity_field = []

        for i in range(len(closest_pts)):
            arrow = Arrow(
                start_pt=Y_track[i, :], end_pt=closest_pts[i, :], c=(255, 0, 0),
            )
            velocity_field.append(arrow)
        try:
            plotter = Plotter()
            plotter.show(Y_true_pc, Y_track_pc, Y_true_line, Y_track_line, closest_pts_pc, velocity_field)
            plotter.interactive().close()
        except KeyboardInterrupt:
            plotter.interactive().close()
            print("Shutting down")
        
if __name__ == "__main__":
    bag = rosbag.Bag('/home/hollydinkel/rmdlo_tracking/src/trackdlo/data/rope_with_marker_stationary_curved.bag')
    rgb_length = bag.get_message_count('/camera/color/image_raw')
    pc_length = bag.get_message_count('/camera/depth/color/points')
    rospy.init_node("evaluator")
    e = TrackDLOEvaluator(pc_length)
    try:
        rospy.spin()
    except:
        print("Shutting down")
