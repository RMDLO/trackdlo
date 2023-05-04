#!/bin/bash
# Loop

# for bag in 0
# do
#     for pct in 0 10 20 30 40 50
#     do
#         for trial in 0 1 2 3 4 5 6 7 8 9
#         do
#             for alg in cdcpd
#             do
#                 terminator -e 'roscore -p 1234' &
#                 first_teminal=$!
#                 terminator -e "cd cdcpd_ws && source devel/setup.bash && python src/cdcpd/ros_nodes/simple_cdcpd_node.py $bag --wait" &
#                 second_teminal=$!
#                 terminator -e "cd rmdlo_tracking && source devel/setup.bash && roslaunch trackdlo evaluation.launch alg:=$alg bag_file:=$bag trial:=$trial pct_occlusion:=$pct --wait" &
#                 third_teminal=$!
#                 sleep 80
#                 rosnode kill -a
#                 killall -9 rosmaster
#                 kill $first_terminal
#                 kill $second_terminal
#                 kill $third_terminal
#             done
#         done
#     done
# done

# for bag in 1 2
# do
#     for pct in 0
#     do
#         for trial in 0 1 2 3 4 5 6 7 8 9
#         do
#             for alg in cdcpd
#             do
#                 terminator -e 'roscore -p 1234' &
#                 first_teminal=$!
#                 terminator -e "cd cdcpd_ws && source devel/setup.bash && python src/cdcpd/ros_nodes/simple_cdcpd_node.py $bag --wait" &
#                 second_teminal=$!
#                 terminator -e "cd rmdlo_tracking && source devel/setup.bash && roslaunch trackdlo evaluation.launch alg:=$alg bag_file:=$bag trial:=$trial pct_occlusion:=$pct --wait" &
#                 third_teminal=$!
#                 sleep 75
#                 rosnode kill -a
#                 killall -9 rosmaster
#                 kill $first_terminal
#                 kill $second_terminal
#                 kill $third_terminal
#             done
#         done
#     done
# done


# # image ONLY
# for bag in 0
# do
#     for pct in 40
#     do
#         for trial in 0
#         do
#             for alg in cdcpd
#             do
#                 terminator -e "cd ~/catkin_ws && source devel/setup.bash && roscore" &
#                 first_teminal=$!
#                 terminator -e "cd ~/catkin_ws && source devel/setup.bash && python3 src/cdcpd/ros_nodes/simple_cdcpd_node.py $bag --wait" &
#                 second_terminal=$!
#                 terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo evaluation.launch alg:=$alg bag_file:=$bag trial:=$trial pct_occlusion:=$pct save_images:=true save_errors:=false --wait" &
#                 third_terminal=$!
#                 sleep 80
#                 rosnode kill -a
#                 killall -9 rosmaster
#                 kill $first_terminal
#                 kill $second_terminal
#                 kill $third_terminal
#             done
#         done
#     done
# done

for bag in 3
do
    for pct in 0
    do
        for trial in 0
        do
            for alg in cdcpd
            do
                terminator -e "cd ~/catkin_ws && source devel/setup.bash && roscore" &
                first_teminal=$!
                terminator -e "cd ~/catkin_ws && source devel/setup.bash && python3 src/cdcpd/ros_nodes/simple_cdcpd_node.py $bag --wait" &
                second_terminal=$!
                terminator -e "cd ~/catkin_ws && source devel/setup.bash && roslaunch trackdlo evaluation.launch alg:=$alg bag_file:=$bag trial:=$trial pct_occlusion:=$pct save_images:=true bag_rate:=0.5 --wait" &
                third_terminal=$!
                sleep 50
                rosnode kill -a
                killall -9 rosmaster
                kill $first_terminal
                kill $second_terminal
                kill $third_terminal
            done
        done
    done
done