#!/bin/bash
# Loop

for kernel in 0 3
do
    for bag in 0 1
    do
        for trial in 0 1 2 3 4 5 6 7 8 9
        do
            terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo trackdlo.launch bag_file:=$bag kernel:=$kernel" &
            first_teminal=$!
            terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo evaluation.launch bag_file:=$bag trial:=$trial pct_occlusion:=40 save_location:=/home/jingyixiang/catkin_ws/src/trackdlo/data/dlo_tracking/kernel_analysis/kernel$kernel/ --wait" &
            second_teminal=$!
            sleep 80
            rosnode kill -a
            killall -9 rosmaster
        done
    done
done