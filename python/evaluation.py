#!/usr/bin/env python3

import numpy as np
import cv2

import rosbag
import rospy
from pylab import *
from ros_numpy import point_cloud2
from ros_numpy import numpify
from sensor_msgs.msg import PointCloud2, Image
import message_filters

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from trackdlo import sort_pts

class TrackDLOEvaluator:
    """Extracts node topic created by the TrackDLO algorithm"""
    def __init__(self):
        # self.bag = rosbag.Bag(bag)
        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.pc_sub = message_filters.Subscriber('/camera/depth/color/points', PointCloud2)
        self.trackdlo_results_sub = message_filters.Subscriber('/results_pc', PointCloud2)
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.pc_sub, self.trackdlo_results_sub], 10)
        self.ts.registerCallback(self.callback)
        self.gt_pub = rospy.Publisher('/gt_pts', PointCloud2, queue_size=10)

    def callback(self, rgb_img, pc, track):
        Y_true, head = self.get_ground_truth_nodes(rgb_img, pc)
        Y_track = self.get_tracking_nodes(track, head)
        self.get_piecewise_curve(Y_true)
        
    def get_ground_truth_nodes(self, rgb_img, pc):
        
        proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                                [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                                [             0.0,              0.0,               1.0, 0.0]])
        head = rgb_img.header

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
        Y_true_msg = PointCloud2()
        pc_list = []
        for i in range(num_blobs):
            x = int(keypoints[i].pt[0])
            y = int(keypoints[i].pt[1])
            new_pt = pc[y, x].tolist() # index pointcloud at blob pixel points, may need to sort these using the trackdlo sort_pts function (imported)
            Y_true[i, 0] = new_pt[0]
            Y_true[i, 1] = new_pt[1]
            Y_true[i, 2] = new_pt[2]
            pc_list.append(np.array(new_pt[0:3]).astype(np.float32))
        pc = np.vstack(pc_list).astype(np.float32).T
        rec_project = np.core.records.fromarrays(pc, 
                                                names='x, y, z',
                                                formats = 'float32, float32, float32')            

        Y_true_msg = point_cloud2.array_to_pointcloud2(rec_project, head.stamp, frame_id='camera_color_optical_frame')
        self.gt_pub.publish(Y_true_msg)
            
        return Y_true, head

    def get_tracking_nodes(self, track, head):
        pc_data = point_cloud2.pointcloud2_to_array(track)
        pc = point_cloud2.get_xyz_points(pc_data)
        return pc

    def get_piecewise_curve(self,Y):
        x = Y[:,0]
        y = Y[:,1]
        n_data = Y.shape[0]
        n_seg = n_data - 1
        dys = np.gradient(y, x)
        rgr = DecisionTreeRegressor(max_leaf_nodes=n_seg)
        rgr.fit(x.reshape(-1, 1), dys.reshape(-1, 1))
        dy_dt = rgr.predict(x.reshape(-1, 1)).flatten()
        y_sl = np.ones(len(x)) * np.nan
        fig, ax0 = plt.subplots(1, 1)
        for i in dy_dt:
            msk = dy_dt == i
            lin_reg = LinearRegression()
            lin_reg.fit(x[msk].reshape(-1, 1), y[msk].reshape(-1, 1))
            y_sl[msk] = lin_reg.predict(x[msk].reshape(-1, 1)).flatten()
            ax0.plot([x[msk][0], x[msk][-1]],
                        [y_sl[msk][0], y_sl[msk][-1]],
                        color='r', zorder=1)
        ax0.scatter(x, y)
        ax0.scatter(x, y_sl, color='g', zorder=5)
        plt.savefig("./data/output/piecewise_linear.png")

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