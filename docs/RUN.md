# Installation and Run Instructions

This guide contains information about required dependencies and how to run the TrackDLO ROS package.

## Minimum Requirements

Installation and execution of TrackDLO was verified with the below dependencies on an Ubuntu 20.04 system with ROS Noetic.

* [ROS Noetic](http://wiki.ros.org/noetic/Installation)
* [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) (Our version: 3.3.7)
* [Point Cloud Library](https://pointclouds.org/) (Our version: 1.10.0)
* [OpenCV](https://opencv.org/releases/)
* [NumPy](https://numpy.org/install/)
* [SciPy](https://scipy.org/install/)
* [scikit-image](https://scikit-image.org/)
* [Pillow](https://pillow.readthedocs.io/en/stable/installation.html)
* [ROS Numpy](https://pypi.org/project/rosnumpy/)

We also provide Docker files for compatibility with other system configurations. Refer to the [DOCKER.md](https://github.com/RMDLO/trackdlo/blob/master/docs/DOCKER.md) for more information on using docker, and see the docker [requirements.txt](https://github.com/RMDLO/trackdlo/blob/master/docker/requirements.txt) file for a list of the tested versions of TrackDLO package dependencies.

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

First, [create a ROS workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace). Next, `cd YOUR_TRACKING_ROS_WORKSPACE/src`. Clone the TrackDLO repository into this workspace and build the package:

```bash
git clone https://github.com/RMDLO/trackdlo.git
catkin build trackdlo
source ../devel/setup.bash
```

All configurable parameters for the TrackDLO algorithm are in [`launch/trackdlo.launch`](https://github.com/RMDLO/trackdlo/blob/master/launch/trackdlo.launch). Rebuilding the package is not required for any parameter modifications to take effect. However, `catkin build` is required after modifying any C++ files. Remember that `source <YOUR_TRACKING_ROS_WS>/devel/setup.bash` is required in every terminal running TrackDLO ROS nodes.

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

Once all parameters in `launch/trackdlo.launch` are set to proper values, run TrackDLO with the following steps:
1. Launch the RGB-D camera node
2. Launch RViz to visualize all published topics: 
```bash
roslaunch trackdlo visualize_output.launch
```
3. Launch the TrackDLO node to publish tracking results:
```bash
roslaunch trackdlo trackdlo.launch
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
roslaunch trackdlo trackdlo.launch
```

## Run TrackDLO with Recorded ROS Bag Data:
1. Download one of the provided rosbag experiment `.bag` files [here](https://doi.org/10.13012/B2IDB-2916472_V1). Note: the file sizes are large! Please first make sure there is enough local storage space on your machine.
2. Open a new terminal and run 
```bash
roslaunch trackdlo visualize_output.launch bag:=True
```
3. In another terminal, run the below command to start the tracking algorithm with parameters for the `stationary.bag`, `perpendicular_motion.bag`, and `parallel_motion.bag` files used for quantitative evaluation in the TrackDLO paper.
```bash
roslaunch trackdlo trackdlo_eval.launch
```
If testing any of the other provided `.bag` files, run the below command:
```bash
roslaunch trackdlo trackdlo.launch
```
4. Open a third ternimal and run the below command to replay the `.bag` file and publish its topics:
```bash
rosbag play --clock /path/to/filename.bag
```
Occlusion can also be injected using our provided `simulate_occlusion_eval.py` script. Run the below command and draw bounding rectangles for the occlusion mask in the graphical user interface that appears:
```bash
rosrun trackdlo simulate_occlusion_eval.py
```

## Data:

The ROS bag files used in our paper and the supplementary video can be found [here](https://doi.org/10.13012/B2IDB-2916472_V1). The data include three scenarios used for the quantitative evaluation shared in the original TrackDLO manuscript; three manipulation scenarios for a rope and rubber tubing shared in the supplementary video; and three failure cases shared in the supplementary video.

### Notes on Running the Bag Files

* The rope and the rubber tubing require different hsv thresholding values. Both of these objects use the `hsv_threshold_upper_limit` default value = `130 255 255` however the rope uses the `hsv_threshold_lower_limit` default value = `90 90 30` and the rubber tubing uses the `hsv_threshold_upper_limit` default value = `100 200 60`.
* For `.bag` files in `rope/`, `rubber_tubing/`, and `failure_cases/`, the `camera_info_topic` is published under `/camera/aligned_depth_to_color/camera_info`. For `.bag` files in `quantitative_eval/`, the `camera_info_topic` is published under `/camera/color/camera_info`. 
