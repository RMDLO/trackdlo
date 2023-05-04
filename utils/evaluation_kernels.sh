#!/bin/bash
# Loop

# individual runs
for kernel in 2
do
    for bag in 5
    do
        for trial in 0
        do
            terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo trackdlo.launch bag_file:=$bag kernel:=$kernel" &
            first_teminal=$!
            terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo evaluation.launch bag_file:=$bag trial:=$trial pct_occlusion:=0 save_location:=/home/jingyixiang/catkin_ws/src/trackdlo/data/dlo_tracking/kernel_comparison/kernel$kernel/ save_images:=true --wait" &
            second_teminal=$!
            sleep 70
            rosnode kill -a
            killall -9 rosmaster
        done
    done
done

# for kernel in 0 1 2 3
# do
#     for bag in 4
#     do
#         for trial in 0
#         do
#             terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo trackdlo.launch bag_file:=$bag kernel:=$kernel" &
#             first_teminal=$!
#             terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo evaluation.launch bag_file:=$bag trial:=$trial pct_occlusion:=0 save_location:=/home/jingyixiang/catkin_ws/src/trackdlo/data/dlo_tracking/kernel_comparison/kernel$kernel/ save_images:=true --wait" &
#             second_teminal=$!
#             sleep 35
#             rosnode kill -a
#             killall -9 rosmaster
#         done
#     done
# done

# for kernel in 0 1 2 3
# do
#     for bag in 5
#     do
#         for trial in 0
#         do
#             terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo trackdlo.launch bag_file:=$bag kernel:=$kernel" &
#             first_teminal=$!
#             terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo evaluation.launch bag_file:=$bag trial:=$trial pct_occlusion:=0 save_location:=/home/jingyixiang/catkin_ws/src/trackdlo/data/dlo_tracking/kernel_comparison/kernel$kernel/ save_images:=true --wait" &
#             second_teminal=$!
#             sleep 70
#             rosnode kill -a
#             killall -9 rosmaster
#         done
#     done
# done

# for kernel in 0 1 2 3
# do
#     for bag in 4
#     do
#         for trial in 1 2 3 4 5 6 7 8 9
#         do
#             terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo trackdlo.launch bag_file:=$bag kernel:=$kernel" &
#             first_teminal=$!
#             terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo evaluation.launch bag_file:=$bag trial:=$trial pct_occlusion:=0 save_location:=/home/jingyixiang/catkin_ws/src/trackdlo/data/dlo_tracking/kernel_comparison/kernel$kernel/ --wait" &
#             second_teminal=$!
#             sleep 35
#             rosnode kill -a
#             killall -9 rosmaster
#         done
#     done
# done

# for kernel in 0 1 2 3
# do
#     for bag in 5
#     do
#         for trial in 1 2 3 4 5 6 7 8 9
#         do
#             terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo trackdlo.launch bag_file:=$bag kernel:=$kernel" &
#             first_teminal=$!
#             terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo evaluation.launch bag_file:=$bag trial:=$trial pct_occlusion:=0 save_location:=/home/jingyixiang/catkin_ws/src/trackdlo/data/dlo_tracking/kernel_comparison/kernel$kernel/ --wait" &
#             second_teminal=$!
#             sleep 70
#             rosnode kill -a
#             killall -9 rosmaster
#         done
#     done
# done