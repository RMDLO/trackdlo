#!/bin/bash
# Loop

# for bag in 0
# do
#     for pct in 0 25 50
#     do
#         for trial in 1 2 3 4 5 6 7 8 9
#         do
#             for alg in cdcpd2
#             do
#                 terminator -e "cd rmdlo_tracking && source devel/setup.bash && roslaunch cdcpd_ros cdcpd2.launch bag_file:=$bag" &
#                 first_teminal=$!
#                 terminator -e "cd rmdlo_tracking && source devel/setup.bash && roslaunch trackdlo evaluation.launch alg:=$alg bag_file:=$bag trial:=$trial pct_occlusion:=$pct --wait" &
#                 second_teminal=$!
#                 sleep 80
#                 terminator -e "rosnode kill -a && killall -9 rosmaster"
#                 # killall -9 rosmaster
#                 kill $first_terminal
#                 kill $second_terminal
#             done
#         done
#     done
# done

for bag in 1 2
do
    for pct in 0
    do
        for trial in 0 1 2 3 4 5 6 7 8 9
        do
            for alg in cdcpd2
            do
                terminator -e "cd rmdlo_tracking && source devel/setup.bash && roslaunch cdcpd cdcpd2.launch bag_file:=$bag" &
                first_teminal=$!
                terminator -e "cd rmdlo_tracking && source devel/setup.bash && roslaunch trackdlo evaluation.launch alg:=$alg bag_file:=$bag trial:=$trial pct_occlusion:=$pct --wait" &
                second_teminal=$!
                sleep 80
                rosnode kill -a
                killall -9 rosmaster
                kill $first_terminal
                kill $second_terminal
            done
        done
    done
done