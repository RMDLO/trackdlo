#!/usr/bin/env python

import rospy
# import ros_numpy
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

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.depth, format = rs.format.z16, framerate = 30, width = 1280, height = 720)
config.enable_stream(rs.stream.color, format = rs.format.bgr8, framerate = 30, width = 1280, height = 720)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
temporal = rs.temporal_filter()

decimate.set_option(rs.option.filter_magnitude, 3.0)
# temporal.set_option(rs.option.filter_option, 1)
temporal.set_option(rs.option.filter_smooth_alpha, 0.2)
temporal.set_option(rs.option.filter_smooth_delta, 100.0)

def align_color():

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_frame = temporal.process(depth_frame)
    depth_frame = decimate.process(depth_frame)

    # Grab new intrinsics (may be changed by decimation)
    depth_intrinsics = rs.video_stream_profile (depth_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    color_intrinsics = rs.video_stream_profile (color_frame.profile).get_intrinsics()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)

    # Pointcloud data to arrays
    v, t = points.get_vertices(), points.get_texture_coordinates()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    # header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'camera_depth_optical_frame'
    pt = []

    # h, w = np.shape(depth_image)
    v, u = (texcoords * (color_intrinsics.width, color_intrinsics.height) + 0.5).astype(np.uint32).T
    # v = v + 20
    np.clip(u, 0, color_intrinsics.height-1, out=u)
    np.clip(v, 0, color_intrinsics.width-1, out=v)

    for i in range (0, h):
        for j in range (0, w):
           
            ux = u[i*w + j]
            uy = v[i*w + j]

            r = color_image[ux, uy, 0]
            g = color_image[ux, uy, 1]
            b = color_image[ux, uy, 2]
            a = 255

            x = verts [i*w + j, 0]
            y = verts [i*w + j, 1]
            z = verts [i*w + j, 2]

            rgb = struct.unpack('I', struct.pack('BBBB', r, g, b, a))[0]

            # only show close points
            if z <= 0.7 and (r < 80 and g < 80 and z < 80):
                pt.append([x, y, z, rgb])

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              # PointField('rgb', 12, PointField.UINT32, 1),]
              PointField('rgba', 12, PointField.UINT32, 1),]

    #create pcl from points
    converted_points = pcl2.create_cloud(header, fields, pt)
    # converted_points = pcl2.create_cloud(header, fields, verts)

    depth_pub.publish(converted_points)
    # end of point cloud publisher

    # begin publishing image
    color_image_converted = cv2.cvtColor (color_image, cv2.COLOR_BGR2RGB)
    img_msg = Image()
    img_msg.header.stamp = rospy.Time.now()
    img_msg.height = color_intrinsics.height
    img_msg.width = color_intrinsics.width
    img_msg.encoding = "rgb8"
    img_msg.is_bigendian = False
    img_msg.step = 3 * color_intrinsics.width
    img_msg.data = color_image_converted.tobytes()
    img_pub.publish(img_msg)

if __name__=='__main__':
    rospy.init_node ('depthInfo')
    depth_pub = rospy.Publisher ('/points1', PointCloud2, queue_size=1)
    img_pub = rospy.Publisher ('/original_img', Image, queue_size=1)

    # tf
    # camera_color_optical_frame -> camera_color_frame

    broadcaster_1 = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped_1 = geometry_msgs.msg.TransformStamped()

    static_transformStamped_1.header.stamp = rospy.Time.now()
    static_transformStamped_1.header.frame_id = "camera_color_optical_frame"
    static_transformStamped_1.child_frame_id = "camera_color_frame"

    quat_1 = [0.5, -0.5, 0.5, 0.5]

    static_transformStamped_1.transform.rotation.x = quat_1[0]
    static_transformStamped_1.transform.rotation.y = quat_1[1]
    static_transformStamped_1.transform.rotation.z = quat_1[2]
    static_transformStamped_1.transform.rotation.w = quat_1[3]


    # camera_color_frame -> camera_link

    broadcaster_2 = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped_2 = geometry_msgs.msg.TransformStamped()

    static_transformStamped_2.header.stamp = rospy.Time.now()
    static_transformStamped_2.header.frame_id = "camera_color_frame"
    static_transformStamped_2.child_frame_id = "camera_link"

    trans_2 = [-0.000351057737134, -0.0148385819048, -0.000117231989861]
    quat_2 = [0.00429561594501, 0.000667857821099, -0.00226634810679, 0.999987959862]

    static_transformStamped_2.transform.translation.x = trans_2[0]
    static_transformStamped_2.transform.translation.y = trans_2[1]
    static_transformStamped_2.transform.translation.z = trans_2[2]

    static_transformStamped_2.transform.rotation.x = quat_2[0]
    static_transformStamped_2.transform.rotation.y = quat_2[1]
    static_transformStamped_2.transform.rotation.z = quat_2[2]
    static_transformStamped_2.transform.rotation.w = quat_2[3]


    # camera_link -> camera_depth_frame

    broadcaster_3 = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped_3 = geometry_msgs.msg.TransformStamped()

    static_transformStamped_3.header.stamp = rospy.Time.now()
    static_transformStamped_3.header.frame_id = "camera_link"
    static_transformStamped_3.child_frame_id = "camera_depth_frame"

    quat_3 = [0, 0, 0, 1.0]

    static_transformStamped_3.transform.rotation.x = quat_3[0]
    static_transformStamped_3.transform.rotation.y = quat_3[1]
    static_transformStamped_3.transform.rotation.z = quat_3[2]
    static_transformStamped_3.transform.rotation.w = quat_3[3]


    # camera_depth_frame -> camera_depth_optical_frame

    broadcaster_4 = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped_4 = geometry_msgs.msg.TransformStamped()

    static_transformStamped_4.header.stamp = rospy.Time.now()
    static_transformStamped_4.header.frame_id = "camera_depth_frame"
    static_transformStamped_4.child_frame_id = "camera_depth_optical_frame"

    quat_4 = [-0.5, 0.5, -0.5, 0.5]

    static_transformStamped_4.transform.rotation.x = quat_4[0]
    static_transformStamped_4.transform.rotation.y = quat_4[1]
    static_transformStamped_4.transform.rotation.z = quat_4[2]
    static_transformStamped_4.transform.rotation.w = quat_4[3]

    while not rospy.is_shutdown():

        static_transformStamped_1.header.stamp = rospy.Time.now()
        broadcaster_1.sendTransform(static_transformStamped_1)

        static_transformStamped_2.header.stamp = rospy.Time.now()
        broadcaster_2.sendTransform(static_transformStamped_2)

        static_transformStamped_3.header.stamp = rospy.Time.now()
        broadcaster_3.sendTransform(static_transformStamped_3)

        static_transformStamped_4.header.stamp = rospy.Time.now()
        broadcaster_4.sendTransform(static_transformStamped_4)

        # depth pub
        start_time = time.time()
        align_color()
        print (time.time() - start_time)

    # Stop streaming
    pipeline.stop()