#!/bin/bash

# Copyright (c) 2023, UNIVERSITY OF ILLINOIS URBANA-CHAMPAIGN. All rights reserved.

# Stop in case of any error.
set -e

# Create catkin workspace.
mkdir -p ${CATKIN_WS}/src
cd ${CATKIN_WS}/src
catkin init
cd ..
catkin build
source devel/setup.bash