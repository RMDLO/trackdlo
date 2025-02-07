#!/bin/bash

# Source ROS distribution and Catkin Workspace
source /opt/ros/noetic/setup.bash
source ${DEVEL}/setup.bash

exec "$@"
