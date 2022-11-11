# TrackDLO Package

***********

**Note: eval files are currently broken, please do not touch**

***********

### To test the most recent version of the tracking algorithm:

1. Run ```roslaunch TrackDLO realsense_node.launch```. This will bring up the rviz window with color image, mask, and tracking result (2D and 3D) visualized.

2. Open a new terminal and run ```rosrun TrackDLO tracking_ros_dev.py```. This will start the tracking algorithm and publish all results.

### To evaluate a certain method:

1. Run ```roslaunch TrackDLO realsense_node_eval.launch```. Use arg ```method:=``` to specify the method to evaluate. Current available options are ```mct_predict``` (ours) and ```cpd_lle```.

### To compare two methods side by side:

1. Simply run ```roslaunch TrackDLO realsense_node_eval_all.launch```. This will start both ```eval_mct_predict.py``` and ```eval_cpd_lle.py```.
