# Installation and Run Instructions

This guide contains information about required dependencies and how to run the TrackDLO ROS package.

## Minimum Requirements

Installation and execution of TrackDLO was verified with the below dependencies.

* [ROS Noetic](http://wiki.ros.org/noetic/Installation)
* [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) (Our version: 3.3.7)
* [Point Cloud Library](https://pointclouds.org/) (Our version: 1.10.0)
* [OpenCV](https://opencv.org/releases/) (Our version: 4.2.0)
* [NumPy](https://numpy.org/install/) (Our version: 1.9.1)
* [SciPy](https://scipy.org/install/) (Our version: 1.23.3)
* [scikit-image](https://scikit-image.org/) (Our version: 0.18.0)
* [Pillow](https://pillow.readthedocs.io/en/stable/installation.html) (Our version: 9.2.0)

## Other Requirements

We used an Intel RealSense d435 camera in all of the experiments performed in our paper. 

* [librealsense](https://github.com/IntelRealSense/librealsense) and [realsense-ros](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy) (for testing with the RealSense D435 camera and the corresponding camera configuration file we provided)

## Installation

The repository is organized into the following directories:

```
  ├── trackdlo/   # contains TrackDLO class and corresponding ROS node
  ├── launch/     # contains ROS launch files for the camera and RViz
  ├── rviz/       # contains Rviz configurations for visualization
  ├── config/     # contains camera configurations
  └── utils/      # contains scripts used for testing and evaluation
```

First, clone the repository into a ROS workspace and build the package:

```bash
$ git clone https://github.com/RMDLO/trackdlo.git
$ catkin build trackdlo
```

All configurable parameters for the TrackDLO algorithm are in `launch/trackdlo.launch`. Rebuilding the package is not required for any parameter modifications to take effect. However, `catkin build` is required after modifying any C++ files.

## Usage

To run TrackDLO, the three ROS topic names below should be modified in `launch/trackdlo.launch` to match the ROS topic names published by the user's RGB-D camera:
* `camera_info_topic`: a CameraInfo topic that contains the camera's projection matrix
* `rgb_topic`: a RGB image stream topic
* `depth_topic`: a depth image stream topic

**Note: TrackDLO assumes the RGB image and its corresponding depth image are ALIGNED and SYNCHRONIZED. This means depth_image(i, j) should contain the depth value of pixel rgb_image(i, j) and the two images should be published with the same ROS timestamp.**

TrackDLO uses color thresholding to obtain the DLO segmentation mask. Below are two different ways to set the color thresholding parameters:
* If the DLO of interest only has one color: you can use the parameters `hsv_threshold_upper/lower_limit` and set their values with format `h_value s_value v_value` (`h_value<space>s_value<space>v_value`).
* If the DLO of interest has multiple colors: set `multi_color_dlo` to `true` in `launch/trackdlo.launch`, then you can modify the function `color_thresholding` in `trackdlo/src/initialize.py` and `trackdlo/src/trackdlo_node.cpp` to customize the DLO segmentation process.

Other useful parameters:
* `num_of_nodes`: the number of nodes initialized for the DLO
* `visualize_initialization_process`: if set to `true`, OpenCV windows will appear to visualize the results of each step in initialization. This is helpful for debugging in the event of initialization failures.

Once all parameters in `trackdlo.launch` are set to proper values, run TrackDLO with the following steps:
1. Launch the RGB-D camera node
2. Launch RViz to visualize all published topics: 
```bash
$ roslaunch trackdlo visualize_output.launch
```
3. Launch the TrackDLO node to publish tracking results:
```bash
$ roslaunch trackdlo trackdlo.launch
```

The TrackDLO node outputs the following:
* `/trackdlo/results_marker`: the tracking result with nodes visualized with spheres and edges visualized with cylinders in MarkerArray format, 
* `/trackdlo/results_pc`: the tracking results in PointCloud2 format
* `/trackdlo/results_img`: the tracking results projected onto the received input RGB image in RGB Image format

## Run TrackDLO with a RealSense D435 camera:
This package was tested using an Intel RealSense D435 camera. The exact camera configurations used are provided in `/config/preset_decimation_4.0_depth_step_100.json` and can be loaded into the camera using the launch files from `realsense-ros`. Run the following commands to start the RealSense camera and the tracking node:
1. Launch an RViz window visualizing the color image, mask, and tracking result (in both the image and the 3D pointcloud) with
```bash
roslaunch trackdlo realsense_node.launch
```
2. Launch the TrackDLO tracking node and publish messages containing the estimated node positions defining the object shape with
```bash
$ roslaunch trackdlo trackdlo.launch
```


## Run TrackDLO with Recorded ROS Bag Data:
1. Download the `.bag` files from [here](https://drive.google.com/drive/folders/1YjX-xfbNfm_G9FYbdw1voYxmd9VA-Aho?usp=sharing) and place them in your ROS workspace.
2. Open a new terminal and run 
```bash
$ roslaunch trackdlo visualize_output.launch
```
3. In another terminal, run the below command to start the tracking algorithm:
```bash
$ roslaunch trackdlo trackdlo.launch
```
4. Open a third ternimal and run the below command to replay the `.bag` file and publish its topics:
```bash
$ rosbag play <name_of_the_bag_file>.bag
```


## Data:

To enable compatibility with different RGB-D camera models, we recently updated the package to work with depth images instead of ordered point clouds. We are working on recollecting the corresponding ROS `.bag` files and will share them shortly. For the time being, the original `.bag` files can still be found [here](https://drive.google.com/drive/folders/1YjX-xfbNfm_G9FYbdw1voYxmd9VA-Aho?usp=sharing).
