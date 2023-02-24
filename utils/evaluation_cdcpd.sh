#!/bin/bash
# Loop

for bag in 0
do
    for pct in 0 25 50 75
    do
        for trial in 0 1 2 3 4 5 6 7 8 9
        do
            for alg in cdcpd
            do
                terminator -e 'roscore -p 1234' &
                first_teminal=$!
                terminator -e 'cd rmdlo_tracking && source devel/setup.bash && python src/cdcpd/ros_nodes/simple_cdcpd_node.py --wait' &
                second_teminal=$!
                terminator -e "cd rmdlo_tracking && source devel/setup.bash && roslaunch trackdlo evaluation.launch alg:=$alg bag_file:=$bag trial:=$trial pct_occlusion:=$pct --wait" &
                third_teminal=$!
                sleep 36
                rosnode kill -a
                killall -9 rosmaster
                kill $first_terminal
                kill $second_terminal
                kill $third_terminal
            done
        done
    done
done

for bag in 1 2
do
    for trial in 0 1 2 3 4 5 6 7 8 9
    do
        for alg in cdcpd
        do
            terminator -e 'roscore -p 1234' &
            terminator -e 'cd rmdlo_tracking && source devel/setup.bash && python src/cdcpd/ros_nodes/simple_cdcpd_node.py --wait' &
            terminator -e "cd rmdlo_tracking && source devel/setup.bash && roslaunch trackdlo evaluation.launch alg:=$alg bag_file:=$bag trial:=$trial pct_occlusion:=0 --wait" &
            sleep 26
            rosnode kill -a
            killall -9 rosmaster
        done
    done
done