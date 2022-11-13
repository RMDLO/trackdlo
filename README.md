# TrackDLO Package

***********

**Note: eval files are currently broken, please do not touch**

***********

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
