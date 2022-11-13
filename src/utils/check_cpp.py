#!/usr/bin/env python

import matplotlib.pyplot as plt
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg

import struct
import time
import cv2
import numpy as np
import math

import time
import pickle as pkl

import message_filters
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from scipy import ndimage
from scipy import interpolate

# assuming Y is sorted
# k -- going left for k indices, going right for k indices. a total of 2k neighbors.
def get_nearest_indices (k, Y, idx):
    if idx - k < 0:
        # use more neighbors from the other side?
        # indices_arr = np.append(np.arange(0, idx, 1), np.arange(idx+1, idx+k+1+np.abs(idx-k)))
        indices_arr = np.append(np.arange(0, idx, 1), np.arange(idx+1, idx+k+1))
        return indices_arr
    elif idx + k >= len(Y):
        last_index = len(Y) - 1
        # use more neighbots from the other side?
        # indices_arr = np.append(np.arange(idx-k-(idx+k-last_index), idx, 1), np.arange(idx+1, last_index+1, 1))
        indices_arr = np.append(np.arange(idx-k, idx, 1), np.arange(idx+1, last_index+1, 1))
        return indices_arr
    else:
        indices_arr = np.append(np.arange(idx-k, idx, 1), np.arange(idx+1, idx+k+1, 1))
        return indices_arr

def calc_LLE_weights (k, X):
    W = np.zeros((len(X), len(X)))
    for i in range (0, len(X)):
        indices = get_nearest_indices(int(k/2), X, i)
        xi, Xi = X[i], X[indices, :]
        print("--- xi ---")
        print(xi)
        print("--- Xi ---")
        print(Xi)

        component = np.full((len(Xi), len(xi)), xi).T - Xi.T
        Gi = np.matmul(component.T, component)
        print("--- Gi ---")
        print(Gi)

        # Gi might be singular when k is large
        if np.linalg.det(Gi) != 0:
            Gi_inv = np.linalg.inv(Gi)
        else:
            print("Gi singular at entry " + str(i))
            epsilon = 0.00000001
            Gi_inv = np.linalg.inv(Gi + epsilon*np.identity(len(Gi)))
        print("--- Gi_inv ---")
        print(Gi_inv)

        wi = np.matmul(Gi_inv, np.ones((len(Xi), 1))) / np.matmul(np.matmul(np.ones(len(Xi),), Gi_inv), np.ones((len(Xi), 1)))
        print("--- wi ---")
        print(wi)
        
        print("--- wi.T ---")
        print(wi.T)
        W[i, indices] = np.squeeze(wi.T)

    return W

if __name__=='__main__':
    m1 = np.arange(0, 18, 1).reshape((6, 3)) / 100
    print(m1)
    out = calc_LLE_weights(2, m1)
    print(out)