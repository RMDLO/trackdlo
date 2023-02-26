import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from labellines import labelLines

algorithms = ['trackdlo', 'cdcpd','cdcpd2', 'gltp']
bags = ['stationary','perpendicular_motion', 'parallel_motion']
duration_frames = [375, 197, 240]
duration_times = [34.1-8, 18.4-5, 22.7-6.5]

# bags = ['stationary','perpendicular_motion']
algorithms_plot = {'trackdlo': 'TrackDLO',
                    'gltp': 'GLTP',
                    'cdcpd': 'CDCPD',
                    'cdcpd2': 'CDCPD2'}
colors = ['cyan','blue','magenta','red']
markers = ['o','^','*','s']
window_size = 10
ROOT_DIR = os.path.abspath(os.curdir)
dir = f'{ROOT_DIR}/src/trackdlo/data'

for n, bag in enumerate(bags):
    if bag=='stationary':
        for pct in [0, 25, 50]:
            ax = plt.gca()
            for i, algorithm in enumerate(algorithms):
                data = []
                for trial in range(0,10):
                    with open(f'{dir}/dlo_tracking/{algorithm}_{trial}_{pct}_{bag}_error.txt', 'r') as file:
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
            plt.xlabel('Time (s)')
            plt.ylabel('Frame Error (mm)')
            plt.ylim(0, 60)
            plt.tight_layout()
            plt.savefig(f'{dir}/frame_error_eval_{bag}_{pct}.png')
            plt.close()

    else:
        for pct in [0]:
            ax = plt.gca()
            for i, algorithm in enumerate(algorithms):
                data = []
                for trial in range(0,10):
                    with open(f'{dir}/dlo_tracking/{algorithm}_{trial}_{pct}_{bag}_error.txt', 'r') as file:
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
            plt.xlabel('Time (s)')
            plt.ylabel('Frame Error (mm)')
            plt.ylim(0, 60)
            plt.tight_layout()
            plt.savefig(f'{dir}/frame_error_eval_{bag}_{pct}.png')
            plt.close()