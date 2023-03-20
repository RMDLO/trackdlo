import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import dirname, abspath, join
import numpy as np
from labellines import labelLines

plt.rcParams.update({'font.size': 15})
titleSize = 22
labelSize = 18

algorithms = ['trackdlo', 'cdcpd2', 'cdcpd2_no_gripper','cdcpd', 'gltp']
bags = ['stationary','perpendicular_motion', 'parallel_motion']
# titles = ['Avg. Error Over 30s vs. $\%$ Occlusion, Stationary DLO', 'Tracking Error vs. Time for Perpendicular Motion', 'Tracking Error vs. Time for Parallel Motion']
titles = ['Stationary', 'Perpendicular Motion', 'Parallel Motion']
duration_frames = [375, 197, 240]
duration_times = [34.1-8, 18.4-5, 22.7-6.5]
pcts = [0, 10, 20, 30, 40, 50]

# bags = ['stationary','perpendicular_motion']
algorithms_plot = {'trackdlo': 'TrackDLO',
                    'gltp': 'GLTP',
                    'cdcpd': 'CDCPD',
                    'cdcpd2': 'CDCPD2',
                    'cdcpd2_no_gripper': 'CDCPD2\nw/o gripper'}
colors = ['red', 'orange', 'deepskyblue', 'b', 'midnightblue']
markers = ['o','^','X','s', 'v']

###################### PLOT TIME VS. FRAME ERROR ######################
window_size = 10
dir = join(dirname(dirname(abspath(__file__))), "data")


for n, bag in enumerate(bags):
    if bag=='stationary':
        continue

        for pct in pcts:
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            for i, algorithm in enumerate(algorithms):
                data = []
                for trial in range(0,10):
                    file_path = f'{dir}/dlo_tracking/algo_comparison/{algorithm}/{bag}/{algorithm}_{trial}_{pct}_{bag}_error.txt'
                    with open(file_path, 'r') as file:
                        content = file.readlines()
                        error = []
                        for line in content:
                            row = line.split()
                            error.append(float(row[1])*1000)

                    data.append(error[:duration_frames[n]])

                mean_data_array = np.asarray(data,dtype=object).mean(axis=0)

                average_smoothed_error = pd.Series(list(mean_data_array)).rolling(window_size).mean()
                std_smoothed_error = pd.Series(list(mean_data_array)).rolling(window_size).std()
                time = np.asarray(average_smoothed_error.index/duration_frames[n]) * duration_times[n] - 0.5

                minus_one_std = average_smoothed_error - std_smoothed_error
                minus_one_std[minus_one_std<0] = 0 # set negative std to 0 so that frame error is always positive
                plus_one_std = average_smoothed_error + std_smoothed_error

                ax.plot(time, average_smoothed_error.values, label=f'{algorithms_plot[algorithm]}', alpha=1.0, color=colors[i], marker=markers[i], markevery=30, markersize=12)
                ax.fill_between(time, minus_one_std.values, plus_one_std.values, alpha=0.2, color=colors[i])

            # labelLines(ax.get_lines(), align=False, zorder=2.5, fontsize=20)
            plt.title('Tracking Error vs. Time for Stationary DLO', fontsize=18)
            plt.xlabel('Time (s)', fontsize=15)
            plt.ylabel('Frame Error (mm)', fontsize=15)

            plt.xlim(0, 26)
            plt.axvspan(5, 26, facecolor='gray', alpha=0.3)

            plt.ylim(0, 50)
            plt.tight_layout()
            # plt.grid()
            plt.legend()
            plt.savefig(f'{dir}/eval_frame_error_{bag}_{pct}.png')
            plt.close()

    else:
        for pct in pcts[:1]:
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            for i, algorithm in enumerate(algorithms):
                data = []
                for trial in range(0,10):
                    file_path = f'{dir}/dlo_tracking/algo_comparison/{algorithm}/{bag}/{algorithm}_{trial}_{pct}_{bag}_error.txt'
                    with open(file_path, 'r') as file:
                        content = file.readlines()
                        error = []
                        for line in content:
                            row = line.split()
                            error.append(float(row[1])*1000)
                    
                    data.append(error[:duration_frames[n]])
                mean_data_array = np.asarray(data,dtype=object).mean(axis=0)

                average_smoothed_error = pd.Series(list(mean_data_array)).rolling(window_size).mean()
                std_smoothed_error = pd.Series(list(mean_data_array)).rolling(window_size).std()
                time = np.asarray(average_smoothed_error.index/duration_frames[n]) * duration_times[n] - 0.5

                minus_one_std = average_smoothed_error - std_smoothed_error
                minus_one_std[minus_one_std<0] = 0 # set negative std to 0 so that frame error is always positive
                plus_one_std = average_smoothed_error + std_smoothed_error

                ax.plot(time,average_smoothed_error.values, label=f'{algorithms_plot[algorithm]}', alpha=1.0, color=colors[i], marker=markers[i], markevery=30, markersize=12)
                ax.fill_between(time, minus_one_std.values, plus_one_std.values, alpha=0.2, color=colors[i])

            # labelLines(ax.get_lines(), align=False, zorder=2.5, fontsize=20)
            plt.title(titles[n], fontsize=titleSize)
            plt.xlabel('Time (s)', fontsize=labelSize)
            plt.ylabel('Frame Error (mm)', fontsize=labelSize)

            if bag == "perpendicular_motion":
                plt.xlim(0, 13)
                plt.axvspan(2.55, 14, facecolor='slategray', alpha=0.2)

            if bag == "parallel_motion":
                plt.xlim(0, 16)
                plt.axvspan(3.5, 10, facecolor='darkslateblue', alpha=0.2)
                plt.axvspan(10, 16, facecolor='slategray', alpha=0.2)

            plt.ylim(0, 50)
            plt.tight_layout()
            # plt.grid()
            plt.legend()
            plt.savefig(f'{dir}/eval_frame_error_{bag}.png')
            plt.close()

###################### PLOT PCT OCCLUSION VS. FINAL FRAME ERROR ######################

for n, bag in enumerate(bags):
    if bag=='stationary':
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        for i, algorithm in enumerate(algorithms):
            avg = []
            std = []
            for pct in pcts:
                data = []
                for trial in range(0,10):
                    file_path = f'{dir}/dlo_tracking/algo_comparison/{algorithm}/{bag}/{algorithm}_{trial}_{pct}_{bag}_error.txt'
                    with open(file_path, 'r') as file:
                        content = file.readlines()
                        error = []
                        for line in content:
                            row = line.split()
                            error.append(float(row[1])*1000)

                    data.append(error[duration_frames[n-1]])
                data = np.asarray(data, dtype=object)
                average_error = data.mean()
                std_error = data.std()

                minus_one_std = average_error - std_error
                if minus_one_std < 0:
                    minus_one_std = 0 # set negative std to 0 so that frame error is always positive
                plus_one_std = average_error + std_error

                avg.append(average_error)
                std.append(std_error)
            ax.plot(pcts, avg, linestyle='--', label=f'{algorithms_plot[algorithm]}', alpha=1.0, color=colors[i], marker=markers[i], markersize=12)
            # ax.errorbar(pcts, avg, yerr=std, linewidth=2, color=colors[i], linestyle='dotted')

        plt.title(titles[n], fontsize=titleSize)
        plt.xlabel('Percentage of Occluded Nodes (%)', fontsize=labelSize)
        plt.ylabel('Final Frame Error (mm)', fontsize=labelSize)
        plt.ylim(0, 50)
        plt.tight_layout()
        # plt.grid()
        plt.legend()
        plt.savefig(f'{dir}/eval_pct_error_{bag}.png')
        plt.close()