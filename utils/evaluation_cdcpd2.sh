#!/bin/bash
# Loop

# make directories

for bag in stationary.bag
do
    for pct in 0 25 50 75
    do
        for trial in 1 2 3 4 5 6 7 8 9 10
        do
            for alg in cdcpd2
            do
                echo "starting roscore"
                terminator -e 'roscore -p 1234' &
                first_teminal=$!
                echo "starting nodes for $alg evaluation" &
                terminator -e "source ~/.bashrc && cd rmdlo_tracking/ && source devel/setup.bash && rosrun trackdlo trackdlo" &
                second_teminal=$!
                terminator -e "source ~/.bashrc && cd rmdlo_tracking/ && source devel/setup.bash && rosrun trackdlo evaluation.py $trial $pct $alg $bag" &
                third_teminal=$!
                terminator -e "source ~/.bashrc && cd rmdlo_tracking/src/trackdlo/data/bags && rosbag play -r 0.05 $bag" &
                fourth_teminal=$!
                sleep 1000
                rosnode kill -a
                killall -9 rosmaster
                kill $first_terminal
                kill $second_terminal
                kill $third_terminal
                kill $fourth_terminal
            done
        done
    done
done

terminator -e "python rmdlo_tracking/src/trackdlo/utils/plot.py" 
last_teminal=$!
sleep 2
kill $last_terminal
