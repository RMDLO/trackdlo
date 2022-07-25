#!/usr/bin/env python

import rospy

import cv2
import numpy as np
import os
from PIL import Image
import json
import pyrealsense2
import ros_numpy
import std_msgs.msg

from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image as SensorImage
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge, CvBridgeError

from ros_numpy import point_cloud2
from ros_numpy import image as ros_im

class Detector:

    def __init__(self):
        self.bridge = CvBridge()
        self.model_pub = rospy.Publisher('/detector/color/detect', SensorImage, queue_size=10)
        self.pc_segment_pub = rospy.Publisher('/detector/pointcloud/segmented', PointCloud2, queue_size=1)
        self.zed = "False"
        if self.zed == "True":
            self.img = np.zeros((640,360))
            self.model_sub = rospy.Subscriber('/zedm/zed_node/left_raw/image_raw_color', SensorImage, self.run_detection)
        else:
            self.img = np.zeros((720,1280))
            self.segmentation = np.zeros((1,720,1280))
            self.cam_sub = rospy.Subscriber('/camera/color/image_raw', SensorImage, self.run_detection)
            # self.pc_sub = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.length_pc)
            # self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', SensorImage, self.segment_pc)
            self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', SensorImage, self.segment_pc)
            self.cam_info = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)
            self.depth_info = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.depth_info_callback)
            self.cam_intrinsics = None
            # self.transform_sub = rospy.Subscriber('/tf_static', TFMessage, self.transform_info_callback)

    def camera_info_callback(self, cameraInfo):
        self.cam_intrinsics = pyrealsense2.intrinsics()
        self.cam_intrinsics.width = cameraInfo.width
        self.cam_intrinsics.height = cameraInfo.height
        self.cam_intrinsics.ppx = cameraInfo.K[2]
        self.cam_intrinsics.ppy = cameraInfo.K[5]
        self.cam_intrinsics.fx = cameraInfo.K[0]
        self.cam_intrinsics.fy = cameraInfo.K[4]
        if cameraInfo.distortion_model == 'plumb_bob':
            self.cam_intrinsics.model = pyrealsense2.distortion.brown_conrady
        elif cameraInfo.distortion_model == 'equidistant':
            self.cam_intrinsics.model = pyrealsense2.distortion.kannala_brandt4
        self.cam_intrinsics.coeffs = [i for i in cameraInfo.D]
        K = cameraInfo.K
        self.K = np.asarray(K).astype(float).reshape((3,3))
        self.invK = np.linalg.inv(self.K)
        self.D = cameraInfo.D
        self.cam_time = cameraInfo.header.stamp
    
    # def transform_info_callback(self,tf):

    def depth_info_callback(self, depthInfo):
        self.depth_time = depthInfo.header.stamp

    def run_detection(self, img):
        try:
            self.img = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
            np_image = self.bridge.imgmsg_to_cv2(img, "rgb8")
        except CvBridgeError as e:
            print("CvBridge could not convert images from realsense to opencv")
        if self.zed == "True":
            np_image = np_image[:,:,:3]
        else:
            np_image = np_image

        self.segmentation = np.array([np_image])
        img_msg = ros_im.numpy_to_image(arr=np_image, encoding='rgb8')

        self.model_pub.publish(img_msg)

    def segment_pc(self, depthMsg):
        # np_depth = ros_numpy.numpify(depthMsg)
        try:
            depth = self.bridge.imgmsg_to_cv2(depthMsg, "passthrough")
            np_depth = np.array(depth)
        except CvBridgeError as e:
            print("CvBridge could not convert images from realsense to opencv")

        print(np.shape(np_depth))

        results = []
        for u in range(240):
            for v in range(424):
                result = self.reproject(u,v,np_depth[u,v],list=True)
                results.append(result)

        pc = results

        # header
        # header = std_msgs.msg.Header()
        # header.stamp = rospy.Time.now()
        # header.frame_id = 'camera_color_optical_frame'

        # fields = [PointField('x', 0, PointField.FLOAT32, 1),
        #           PointField('y', 4, PointField.FLOAT32, 1),
        #           PointField('z', 8, PointField.FLOAT32, 1)]

        # header.stamp = rospy.Time.now()
        # pc_msg = pcl2.create_cloud(header, fields, pc)

        rec_project = np.core.records.fromarrays(pc, 
                                            names='x, y, z, r, g, b',
                                            formats = 'float32, float32, float32, float32, float32, float32')                                         

        pc_msg2 = point_cloud2.merge_rgb_fields(rec_project)
        
        pc_msg = point_cloud2.array_to_pointcloud2(pc_msg2, stamp=self.depth_time, frame_id='camera_color_optical_frame')

        # self.pc_segment_pub.publish(pc_msg)

        self.pc_segment_pub.publish(pc_msg)
        
    def reproject(self,u,v,d,list=False):
        '''
        Alignment is not quite right between the segmented pointcloud and the raw textured pointcloud. 
        List = False alignment is better, maybe because it considers the camera distortion model.
        '''
        r = self.img[u,v,0]
        g = self.img[u,v,1]
        b = self.img[u,v,2]
        x_ = (v - self.cam_intrinsics.ppx)/self.cam_intrinsics.fx
        y_ = (u - self.cam_intrinsics.ppy)/self.cam_intrinsics.fy
        z = d/1000.0
        x = x_ * z
        y = y_ * z
        return np.array([x, y, z, r, g, b])

if __name__=='__main__':
    rospy.init_node('detector', anonymous=True)
    d = Detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")