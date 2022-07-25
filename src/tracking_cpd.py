#!/usr/bin/env python

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg

import math
import struct
import time
import cv2
import numpy as np
import pyrealsense2 as rs

import time
import tf2_ros
import geometry_msgs.msg
import pickle as pkl

import message_filters
from pycpd import RigidRegistration
from pycpd import AffineRegistration
from pycpd import DeformableRegistration

def pt2pt_dis_sq(pt1, pt2):
    return np.sum(np.square(pt1 - pt2))

def register(pts, N, mu=0, max_iter=40):

    # initial guess
    y = pts.copy()
    x = np.vstack((np.arange(0, 0.1, (0.1/N)), np.zeros(N), np.zeros(N))).T
    if len(pts[0]) == 2:
        x = np.vstack((np.arange(0, 0.1, (0.1/N)), np.zeros(N))).T
    s = 1
    M = len(pts)
    D = len(pts[0])

    def get_estimates (x, s):
        start_time = time.time()

        new_x = []
        s_top = 0
        s_bottom = 0

        for n_outer in range(0, N):
            xn_top = 0
            xn_bottom = 0
            new_xn = 0
            for m in range(0, M):
                p_n_given_ym = 0
                denominator_sum = (2.0*np.pi*s)**(D/2.0)*mu*N / ((1-mu)*M)

                # for n_inner in range(0, N):
                #     denominator_sum += np.exp(-pt2pt_dis_sq(y[m], x[n_inner]) / (2.0*s))
                y_m_arr = np.full((len(x), D), y[m].copy())
                denominator_sum = np.sum( np.exp( np.sum(np.square(x - y_m_arr), axis=1) * (-0.5 / s) ) )

                p_n_given_ym = np.exp(-pt2pt_dis_sq(y[m], x[n_outer]) / (2*s)) / denominator_sum
                # test
                xn_top += p_n_given_ym*y[m]
                xn_bottom += p_n_given_ym

                s_top += p_n_given_ym*pt2pt_dis_sq(y[m], x[n_outer])
                s_bottom += p_n_given_ym*D
            
            new_xn = xn_top / xn_bottom
            new_x.append(new_xn)
        
        new_s = s_top / s_bottom
        
        print(time.time() - start_time)
        return np.array(new_x), new_s

    prev_x, prev_s = x, s
    new_x, new_s = get_estimates(prev_x, prev_s)
    # it = 0
    tol = 0.0
    
    for it in range (max_iter):
        print(it)
        prev_x, prev_s = new_x, new_s
        new_x, new_s = get_estimates(prev_x, prev_s)

    # print(repr(new_x), new_s)
    return new_x, new_s

saved = False
initialized = False
init_nodes = []
nodes = []
def callback (rgb, depth, pc):
    global saved
    global initialized
    global init_nodes
    global nodes

    # process rgb image
    cur_image = ros_numpy.numpify(rgb)
    # cur_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_BGR2RGB)

    # process depth image
    cur_depth = ros_numpy.numpify(depth)

    # process point cloud
    pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc)
    cur_pc = ros_numpy.point_cloud2.get_xyz_points(pc_data)
    cur_pc = cur_pc.reshape((720, 1280, 3))

    # test
    # print(np.shape(cur_image), np.shape(cur_depth), np.shape(cur_pc))

    # color thresholding
    lower = (0, 0, 135)
    upper = (125, 125, 255)
    mask = cv2.inRange(cur_image, lower, upper)
    mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    mask = (mask/255).astype(int)

    filtered_pc = cur_pc*mask
    filtered_pc = filtered_pc[((filtered_pc[:, :, 0] != 0) | (filtered_pc[:, :, 1] != 0) | (filtered_pc[:, :, 2] != 0))]
    filtered_pc = filtered_pc[filtered_pc[:, 2] < 0.605]
    filtered_pc = filtered_pc[filtered_pc[:, 2] > 0.5]

    # # save points
    # if not saved:
    #     username = 'ablcts18'
    #     folder = 'tracking/'
    #     f = open("/home/" + username + "/Research/" + folder + "ros_pc.json", 'wb')
    #     pkl.dump(filtered_pc, f)
    #     f.close()
    #     saved = True

    # downsample to 2.5%
    filtered_pc = filtered_pc[::int(1/0.1)]
    # add color
    pc_rgba = struct.unpack('I', struct.pack('BBBB', 255, 0, 0, 255))[0]
    pc_rgba_arr = np.full((len(filtered_pc), 1), pc_rgba)
    filtered_pc_colored = np.hstack((filtered_pc, pc_rgba_arr)).astype('O')
    filtered_pc_colored[:, 3] = filtered_pc_colored[:, 3].astype(int)

    print(np.shape(filtered_pc_colored), filtered_pc_colored[0, 3])

    # filtered_pc = filtered_pc.reshape((len(filtered_pc)*len(filtered_pc[0]), 3))
    header.stamp = rospy.Time.now()
    converted_points = pcl2.create_cloud(header, fields, filtered_pc_colored)
    pc_pub.publish(converted_points)

    # register nodes
    if not initialized:
        init_nodes, _ = register(filtered_pc, 30, mu=0, max_iter=40)
        initialized = True
        # header.stamp = rospy.Time.now()
        # converted_init_nodes = pcl2.create_cloud(header, fields, init_nodes)
        # init_nodes_pub.publish(converted_init_nodes)

    # cpd
    if initialized:
        reg = DeformableRegistration(**{'X': filtered_pc, 'Y': init_nodes, 'w': 0.05})
        print("finished reg")
        nodes, _ = reg.register()
        init_nodes = nodes

        # add color
        nodes_rgba = struct.unpack('I', struct.pack('BBBB', 0, 0, 0, 255))[0]
        nodes_rgba_arr = np.full((len(nodes), 1), nodes_rgba)
        nodes_colored = np.hstack((nodes, nodes_rgba_arr)).astype('O')
        nodes_colored[:, 3] = nodes_colored[:, 3].astype(int)
        header.stamp = rospy.Time.now()
        converted_nodes = pcl2.create_cloud(header, fields, nodes_colored)
        nodes_pub.publish(converted_nodes)


if __name__=='__main__':
    rospy.init_node('test', anonymous=True)

    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
    pc_sub = message_filters.Subscriber('/camera/depth/color/points', PointCloud2)

    # header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'camera_color_optical_frame'
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgba', 12, PointField.UINT32, 1)]
    pc_pub = rospy.Publisher ('/pts', PointCloud2, queue_size=10)
    init_nodes_pub = rospy.Publisher ('/init_nodes', PointCloud2, queue_size=10)
    nodes_pub = rospy.Publisher ('/nodes', PointCloud2, queue_size=10)

    ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub, pc_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()