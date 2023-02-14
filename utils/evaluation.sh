#!/bin/bash
# Loop

for pct in 25 50 75
do
    for trial in 1 2 3 4 5 6 7 8 9 10
    do
        for alg in trackdlo gltp
        do
            echo "starting roscore"
            terminator -e 'roscore' &
            first_teminal=$!
            echo "starting nodes for $alg evaluation" &
            terminator -e "cd rmdlo_tracking/ && source devel/setup.bash && rosrun trackdlo trackdlo.py $alg" &
            second_teminal=$!
            terminator -e "cd rmdlo_tracking/ && source devel/setup.bash && rosrun trackdlo evaluation.py $trial $pct $alg" &
            third_teminal=$!
            terminator -e 'sleep 2; cd rmdlo_tracking/ && rosbag play -r 0.1 src/trackdlo/data/rope_with_marker_stationary_curved.bag'
            fourth_teminal=$!
            sleep 600
            killall -9 rosmaster
            kill $first_terminal
            kill $second_terminal
            kill $third_terminal
            kill $fourth_terminal
        done
        for alg in cdcpd
        do
            echo "starting roscore"
            terminator -e 'roscore' &
            first_teminal=$!
            echo "starting nodes for $alg evaluation" &
            terminator -e "cd rmdlo_tracking/ && source devel/setup.bash && python src/cdcpd/simple_cpd_node.py" &
            second_teminal=$!
            terminator -e "cd rmdlo_tracking/ && source devel/setup.bash && rosrun trackdlo evaluation.py $trial $pct $alg" &
            third_teminal=$!
            terminator -e 'sleep 2; cd rmdlo_tracking/ && rosbag play -r 0.1 src/trackdlo/data/rope_with_marker_stationary_curved.bag'
            fourth_teminal=$!
            sleep 600
            killall -9 rosmaster
            kill $first_terminal
            kill $second_terminal
            kill $third_terminal
            kill $fourth_terminal
        done
    done
done

terminator -e "python rmdlo_tracking/src/trackdlo/utils/plot.py" 
last_teminal=$!
sleep 2
kill $last_terminal
