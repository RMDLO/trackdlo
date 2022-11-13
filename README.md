# TrackDLO ROS Package

This is the TrackDLO ROS package for tracking deformable linear objects under occlusion. The TrackDLO algorithm solves the problem of real-time state estimation of Deformable Linear Objects (DLOs), like wires and ropes, under occlusion. The goal of DLO state estimation under occlusion is to extract the structure of the DLO from a noisy or incomplete set of measurements. TrackDLO accounts for directional rigidity to infer the motion of the occluded part of the object from the motion of the visible part. TrackDLO also introduces the notion of a geodesic proximity metric for linking a set of nodes which represent the configuration of the DLO. This modified proximity metric not only improves tracking under occlusion, but also mitigates tracking entanglement for cases of self-occlusion. TrackDLO performs robust wire state estimation under known confounders like partial occlusion by other objects, tip occlusion, and self-occlusion.

<p align="center">
  <img src="images/ours.png" width="500" title="hover text">
</p>

### To test the most recent version of TrackDLO with RGB-D camera stream:

1. Run ```roslaunch TrackDLO realsense_node.launch```. This will bring up the rviz window with color image, mask, and tracking result (2D and 3D) visualized.

2. Open a new terminal and run ```rosrun TrackDLO tracking_ros_dev.py```. This will start the tracking algorithm and publish all results.

### To test the most recent version of TrackDLO with ROS bag files:

1. Download the bag files from [here](https://drive.google.com/drive/folders/1AwMXysdzRQLz7w8umj66rrKa-Bh0XlVJ?usp=share_link) and place them in your ROS workspace.
2. Run ```roslaunch TrackDLO replay_bag.launch```. This will bring up the rviz window with color image, mask, and tracking result (2D and 3D) visualized. The RGB-D camera node will not be started.
3. Open a new terminal and run ```rosrun TrackDLO track_from_bag_replay.py```. This will start the tracking algorithm and the results will be published after the bag file starts running. Note: this script calls functions from ```tracking_ros_dev.py```.
4. Open a new terminal and run ```rosbag play <name_of_the_bag_file>.bag```. This will replay the bag file.

### To evaluate a certain method:

1. Run ```roslaunch TrackDLO realsense_node_eval.launch```. Use arg ```method:=``` to specify the method to evaluate. Current available options are ```mct_predict``` (ours) and ```cpd_lle```.

### To compare two methods side by side:

1. Simply run ```roslaunch TrackDLO realsense_node_eval_all.launch```. This will start both ```eval_mct_predict.py``` and ```eval_cpd_lle.py```.

***********

**Note: eval files are currently broken, please do not touch**

***********
