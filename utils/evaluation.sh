#!/bin/bash
# Loop

for pct in 25 50 75
do
    for trial in 1 2 3 4 5 6 7 8 9 10
    do
        for alg in trackdlo gltp
        do
            terminator -e 'roscore' &
            terminator -e "cd rmdlo_tracking/ && source devel/setup.bash && rosrun trackdlo trackdlo.py $alg" &
            terminator -e "cd rmdlo_tracking/ && source devel/setup.bash && rosrun trackdlo evaluation.py $trial $pct $alg" &
            terminator -e 'cd rmdlo_tracking/ && rosbag play -r 0.1 src/trackdlo/data/rope_with_marker_stationary_curved.bag'
            wait
            rosnode kill -a
            killall -9 roscore
            killall -9 rosmaster
        done
        for alg in cdcpd
        do
            terminator -e 'roscore' &
            terminator -e "cd rmdlo_tracking/ && source devel/setup.bash && python src/cdcpd/simple_cpd_node.py" &
            terminator -e "cd rmdlo_tracking/ && source devel/setup.bash && rosrun trackdlo evaluation.py $trial $pct $alg" &
            terminator -e 'cd rmdlo_tracking/ && rosbag play -r 0.1 src/trackdlo/data/rope_with_marker_stationary_curved.bag'
            wait
            rosnode kill -a
            killall -9 roscore
            killall -9 rosmaster
        done
    done
done