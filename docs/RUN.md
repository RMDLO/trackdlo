# Installation and Run Instructions

This guide contains information about required dependencies and how to run the TrackDLO ROS package.

## Minimum Requirements

Installation and execution of TrackDLO was verified with the below dependencies on an Ubuntu 20.04 system with ROS Noetic.

* [ROS Noetic](http://wiki.ros.org/noetic/Installation)
* [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) (Our version: 3.3.7)
* [Point Cloud Library](https://pointclouds.org/) (Our version: 1.10.0)
* [OpenCV](https://opencv.org/releases/) (Our version: 4.2.0)
* [NumPy](https://numpy.org/install/) (Our version: 1.23.3)
* [SciPy](https://scipy.org/install/) (Our version: 1.9.1)
* [scikit-image](https://scikit-image.org/) (Our version: 0.18.0)
* [Pillow](https://pillow.readthedocs.io/en/stable/installation.html) (Our version: 9.2.0)
* [ROS Numpy](https://pypi.org/project/rosnumpy/) (Our version: 0.0.5)

We also provide Docker files for compatibility with other system configurations, refer to [DOCKER.md](https://github.com/RMDLO/trackdlo/blob/master/docs/DOCKER.md) for more information.

## Other Requirements

We used an Intel RealSense d435 camera in all of the experiments performed in our paper. We used the [librealsense](https://github.com/IntelRealSense/librealsense) and [realsense-ros](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy) packages for testing with the RealSense D435 camera and for obtaining the [camera configuration file](https://github.com/RMDLO/trackdlo/blob/master/config/preset_decimation_4.0_depth_step_100.json).

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
$ cd YOUR_ROS_WORKSPACE/src
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

TrackDLO uses color thresholding in Hue, Saturation, and Value (HSV) color space to obtain the DLO segmentation mask. A tutorial on how to obtain the HSV limits is provided in [`COLOR_THRESHOLD.md`](https://github.com/RMDLO/trackdlo/blob/master/docs/COLOR_THRESHOLD.md)

Other useful parameters:
* `num_of_nodes`: the number of nodes initialized for the DLO
* `result_frame_id`: the tf frame the tracking results (point cloud and marker array) will be published to
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
1. Download the experiment data from [here](https://drive.google.com/file/d/1C7uM515fHXnbsEyx5X38xZUXzBI99mxg/view?usp=drive_link). After unzipping, place the `.bag` files in your ROS workspace. Note: the files are quite large! After unzipping, the bag files will take up around 120 GB of space in total.
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

The ROS bag files used in our paper and the supplementary video can be found [here](https://drive.google.com/file/d/1C7uM515fHXnbsEyx5X38xZUXzBI99mxg/view?usp=drive_link). The `experiment` folder is organized into the following directories:

```
  ├── rope/                # contains the three experiments performed with a rope in the supplementary video
  ├── rubber_tubing/       # contains the three experiments performed with a rope in the supplementary video
  ├── failure_cases/       # contains the three failure cases shown in the supplementary video
  └── quantitative_eval/   # contains the bag files used for quantitative evaluation in our paper
```

### Notes on Running the Bag Files

* The rope and the rubber tubing require different hsv thresholding values. Both of them have hsv upper limit of `130 255 255`, however the rope has hsv lower limit `90 90 30` and the tubing has hsv lower limit `100 200 60`.
* For bag files in `rope/`, `rubber_tubing/`, and `failure_cases/`, the camera info is published under topic `/camera/aligned_depth_to_color/camera_info`. For bag files in `quantitative_eval/`, the camera info is published under `/camera/color/camera_info`.
