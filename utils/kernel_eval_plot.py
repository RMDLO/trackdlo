import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import dirname, abspath, join
import numpy as np
from labellines import labelLines

kernels = ['kernel0', 'kernel1', 'kernel3']
bags = ['stationary','perpendicular_motion']
titles = ['Stationary', 'Perpendicular Motion']
duration_frames = [375, 197, 240]
duration_times = [34.1-8, 18.4-5, 22.7-6.5]
pcts = [0, 10, 20, 30, 40, 50]

# bags = ['stationary','perpendicular_motion']
algorithms_plot = {'kernel0': 'Laplacian',
                    'kernel1': '2nd Order',
                    'kernel3': 'Gaussian'}
colors = ['midnightblue', 'deepskyblue', 'b']
# markers = ['o','^','*','s']
markers = ['o','^','s']

###################### PLOT TIME VS. FRAME ERROR ######################
window_size = 10
ROOT_DIR = abspath(os.curdir)
dir = join(dirname(dirname(abspath(__file__))), "data")

for n, bag in enumerate(bags):
    plt.figure(figsize=(15, 5))
    for i, kernel in enumerate(kernels):  
        ax = plt.gca()
        data = []
        for trial in range(0,10):
            file_path = f'{dir}/dlo_tracking/kernel_analysis/{kernel}/trackdlo_{trial}_40_{bag}_error.txt'
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
        time = np.asarray(average_smoothed_error.index/duration_frames[n]) * duration_times[n]

        minus_one_std = average_smoothed_error - std_smoothed_error
        minus_one_std[minus_one_std<0] = 0 # set negative std to 0 so that frame error is always positive
        plus_one_std = average_smoothed_error + std_smoothed_error

        ax.plot(time,average_smoothed_error.values, label=f'{algorithms_plot[kernel]}', alpha=1.0, color=colors[i], marker=markers[i], markevery=30, markersize=12)
        ax.fill_between(time, minus_one_std.values, plus_one_std.values, alpha=0.2, color=colors[i])

    # labelLines(ax.get_lines(), align=False, zorder=2.5, fontsize=30)
    # plt.title(titles[n])
    plt.xlabel('Time (s)')
    plt.ylabel('Frame Error (mm)')
    plt.ylim(0, 12)
    plt.legend(fontsize=25, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{dir}/eval_frame_error_{bag}.png')
    plt.close()