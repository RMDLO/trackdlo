import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from labellines import labelLines

algorithms = ['trackdlo', 'cdcpd','cdcpd2', 'gltp']
bags = ['stationary','perpendicular_motion', 'parallel_motion']
titles = ['Stationary', 'Perpendicular Motion', 'Parallel Motion']
duration_frames = [375, 197, 240]
duration_times = [34.1-8, 18.4-5, 22.7-6.5]
pcts = [0, 25, 50]

# bags = ['stationary','perpendicular_motion']
algorithms_plot = {'trackdlo': 'TrackDLO',
                    'gltp': 'GLTP',
                    'cdcpd': 'CDCPD',
                    'cdcpd2': 'CDCPD2'}
colors = ['cyan','blue','magenta','red']
markers = ['o','^','*','s']

###################### PLOT TIME VS. FRAME ERROR ######################
window_size = 10
ROOT_DIR = os.path.abspath(os.curdir)
dir = f'{ROOT_DIR}/src/trackdlo/data'

for n, bag in enumerate(bags):
    if bag=='stationary':
        for pct in pcts:
            ax = plt.gca()
            for i, algorithm in enumerate(algorithms):
                data = []
                for trial in range(0,10):
                    with open(f'{dir}/archive_2/{algorithm}_{trial}_{pct}_{bag}_error.txt', 'r') as file:
                        content = file.readlines()
                        error = []
                        for line in content:
                            row = line.split()
                            error.append(float(row[1])*1000)

                    data.append(error[:duration_frames[n]])
                
                mean_data_array = np.asarray(data,dtype=object).mean(axis=0)

                average_smoothed_error = pd.Series(list(mean_data_array)).rolling(window_size).mean()
                std_smoothed_error = pd.Series(list(mean_data_array)).rolling(window_size).std()
                time = np.asarray(average_smoothed_error.index/duration_frames[n]) * duration_times[n]

                minus_one_std = average_smoothed_error - std_smoothed_error
                minus_one_std[minus_one_std<0] = 0 # set negative std to 0 so that frame error is always positive
                plus_one_std = average_smoothed_error + std_smoothed_error

                ax.plot(time, average_smoothed_error.values, label=f'{algorithms_plot[algorithm]}', alpha=1.0, color=colors[i], marker=markers[i], markevery=30, markersize=12)
                ax.fill_between(time, minus_one_std.values, plus_one_std.values, alpha=0.2, color=colors[i])

            labelLines(ax.get_lines(), align=False, zorder=2.5, fontsize=20)
            plt.title(titles[n])
            plt.xlabel('Time (s)')
            plt.ylabel('Frame Error (mm)')
            plt.ylim(0, 50)
            plt.tight_layout()
            plt.savefig(f'{dir}/eval_frame_error_{bag}_{pct}.png')
            plt.close()

    else:
        for pct in pcts[:1]:
            ax = plt.gca()
            for i, algorithm in enumerate(algorithms):
                data = []
                for trial in range(0,10):
                    with open(f'{dir}/archive_2/{algorithm}_{trial}_{pct}_{bag}_error.txt', 'r') as file:
                        content = file.readlines()
                        error = []
                        for line in content:
                            row = line.split()
                            error.append(float(row[1])*1000)

                    data.append(error[:duration_frames[n]])
                
                mean_data_array = np.asarray(data,dtype=object).mean(axis=0)

                average_smoothed_error = pd.Series(list(mean_data_array)).rolling(window_size).mean()
                std_smoothed_error = pd.Series(list(mean_data_array)).rolling(window_size).std()
                time = np.asarray(average_smoothed_error.index/duration_frames[n]) * duration_times[n]

                minus_one_std = average_smoothed_error - std_smoothed_error
                minus_one_std[minus_one_std<0] = 0 # set negative std to 0 so that frame error is always positive
                plus_one_std = average_smoothed_error + std_smoothed_error

                ax.plot(time,average_smoothed_error.values, label=f'{algorithms_plot[algorithm]}', alpha=1.0, color=colors[i], marker=markers[i], markevery=30, markersize=12)
                ax.fill_between(time, minus_one_std.values, plus_one_std.values, alpha=0.2, color=colors[i])

            labelLines(ax.get_lines(), align=False, zorder=2.5, fontsize=20)
            plt.title(titles[n])
            plt.xlabel('Time (s)')
            plt.ylabel('Frame Error (mm)')
            plt.ylim(0, 50)
            plt.tight_layout()
            plt.savefig(f'{dir}/eval_frame_error_{bag}.png')
            plt.close()

###################### PLOT PCT OCCLUSION VS. FINAL FRAME ERROR ######################

for n, bag in enumerate(bags):
    if bag=='stationary':
        ax = plt.gca()
        for i, algorithm in enumerate(algorithms):
            avg = []
            std = []
            for pct in pcts:
                data = []
                for trial in range(0,10):
                    with open(f'{dir}/archive_2/{algorithm}_{trial}_{pct}_{bag}_error.txt', 'r') as file:
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
            print(avg, std, pcts)
            ax.plot(pcts, avg, label=f'{algorithms_plot[algorithm]}', linewidth=4, alpha=1.0, color=colors[i], marker=markers[i], markersize=12)
            # ax.errorbar(pcts, avg, yerr=std, linewidth=10, color=colors[i])

        plt.xlabel('Percentage of Occluded Nodes (%)')
        plt.title(titles[n])
        plt.ylabel('Final Frame Error (mm)')
        plt.ylim(0, 50)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'{dir}/eval_pct_error_{bag}.png')
        plt.close()