# TrackDLO ROS Package

This is the implementation of *TrackDLO: Tracking Deformable Linear Objects Under Occlusion with Motion Coherence* (under review) by Jingyi Xiang, Holly Dinkel, Harry Zhao, Naixiang Gao, Brian Coltin, Trey Smith, and Timothy Bretl. We provide the implementation in C++.

## Overview
The TrackDLO algorithm estimates the shape of a Deformable Linear Object (DLO) under occlusion from a sequence of RGB-D images, for use in manipulation tasks. TrackDLO runs in real-time and requires no physics simulation or gripper movement information. The algorithm improves on previous approaches by addressing three common scenarios which cause their failure: tip occlusion, mid-section occlusion, and self-occlusion.

<p align="center">
  <img src="images/trackdlo1.gif" width="400" title="TrackDLO"> <img src="images/trackdlo2.gif" width="400" title="TrackDLO">
</p>

We also adapt the algorithm introduced in [Deformable One-Dimensional Object Detection for Routing and Manipulation](https://ieeexplore.ieee.org/abstract/document/9697357) and relax the assumption about the DLO initial state to allow complicated setups such as self-crossing and minor occlusion:
<p align="center">
  <img src="images/trackdlo3.gif" width="800" title="TrackDLO initialization">
</p>

## Minimum Requirements
* [ROS Noetic](http://wiki.ros.org/noetic/Installation)
* [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) (Our version: 3.3.7)
* [Point Cloud Library](https://pointclouds.org/) (Our version: 1.10.0)
* [OpenCV](https://opencv.org/releases/) (Our version: 4.2.0)
* [Numpy](https://numpy.org/install/) (Our version: 1.9.1)
* [Scipy](https://scipy.org/install/) (Our version: 1.23.3)

## Other Requirements
* [librealsense](https://github.com/IntelRealSense/librealsense) and [realsense-ros](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy) (for testing with RGB-D camera stream)

## Usage

First, clone the repository into a ROS workspace and build the package:
```bash
git clone https://github.com/RMDLO/trackdlo.git
catkin build
```
All parameters for the TrackDLO algorithm are configurable in `launch/trackdlo.launch`. Rebuilding the package is not required for the parameter modifications to take effect. However, `catkin build` is required after modifying any C++ files.


## Test TrackDLO with your own setup:
To run TrackDLO, three ROS topic names are required in `launch/trackdlo.launch`. You should change their values to match the ROS topic names published by your RGB-D camera:
* `camera_info_topic`: a CameraInfo topic that contains the camera's projection matrix
* `rgb_topic`: a RGB image stream topic
* `depth_topic`: a depth image stream topic

This package uses color thresholding to obtain the DLO segmentation mask. Below are two different ways to set the color thresholding parameters:
* If the DLO of interest only has one color: you can use the parameters `hsv_threshold_upper/lower_limit` and set their values with format `h_value s_value v_value` (`h_value<space>s_value<space>v_value`)
* If the DLO of interest has multiple colors: set `multi_color_dlo` to `true` in `launch/trackdlo.launch`, then you can modify the function `color_thresholding` in `trackdlo/src/initialize.py` and `trackdlo/src/trackdlo_node.cpp` to customize the DLO segmentation process

Other useful parameters:
* `num_of_nodes`: the number of nodes initialized for the DLO
* `visualize_initialization_process`: if set to `true`, OpenCV windows will show up to visualize the results of each step in initialization (helpful for debugging if the initialization process keeps failing)

Once all parameters in `trackdlo.launch` are set to proper values, you can run TrackDLO with the following steps:
1. Launch your RGB-D camera node
2. Launch the rviz window for visualizing results: `roslaunch trackdlo visualize_output.launch`
3. Launcch the TrackDLO node: `roslaunch trackdlo trackdlo.launch`

The TrackDLO node outputs the following:
* `/trackdlo/results_marker`: the tracking results in MarkerArray format, with nodes visualized with spheres and edges visualized with cylinders
* `/trackdlo/results_pc`: the tracking results in PointCloud2 format
* `/trackdlo/tracking_img`: an RGB image with tracking results projected onto the received input RGB image

## Test TrackDLO with a Realsense D435 camera:
This package has been tested with a Intel RealSense D435 camera. The exact camera configurations used are provided in `/config/preset_decimation_4.0_depth_step_100.json` and can be loaded into the camera using the launch files from `realsense-ros`. Run the following commands to start the realsense camera and the tracking node:
1. Run ```roslaunch trackdlo realsense_node.launch```. This will bring up an RViz window visualizing the color image, mask, and tracking result (in both the image and the 3D pointcloud).
2. Open a new terminal and run ```roslaunch trackdlo trackdlo.launch```. This will start the TrackDLO algorithm and publish messages containing the estimated node positions defining the object shape.


## Test TrackDLO with Recorded ROS Bag Data:
1. Download the bag files from [here](https://drive.google.com/drive/folders/1YjX-xfbNfm_G9FYbdw1voYxmd9VA-Aho?usp=sharing) and place them in your ROS workspace.
2. Open a new terminal and run ```roslaunch trackdlo visualize_output.launch```.
3. In another terminal, run ```roslaunch trackdlo trackdlo.launch```. This will start the tracking algorithm.
4. Finally, open another ternimal and run ```rosbag play <name_of_the_bag_file>.bag```. This will replay the bag file and all results will be published in rviz.


## Data:

To allow usage with different types of RGB-D cameras, we recently updated the package to work with depth images instead of ordered point clouds. We are working on recollecting the corresponding ROS bag files and will share them shortly. At the mean time, the old bag files can still be found [here](https://drive.google.com/drive/folders/1YjX-xfbNfm_G9FYbdw1voYxmd9VA-Aho?usp=sharing).
