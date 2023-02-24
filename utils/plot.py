import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from labellines import labelLines

# , 'gltp', 'cdcpd'
algorithms = ['trackdlo', 'gltp', 'cdcpd2']
bags = ['stationary']
algorithms_plot = {'trackdlo': 'TrackDLO',
                    'gltp': 'GLTP',
                    'cdcpd': 'CDCPD',
                    'cdcpd2': 'CDCPD2'}
colors = ['g','b','c','r']
window_size = 10
ROOT_DIR = os.path.abspath(os.curdir)

ax = plt.gca()
for i, algorithm in enumerate(algorithms):
    for bag in bags:
        time = []
        error = []
        with open(f'{ROOT_DIR}/data/dlo_tracking_eval/{algorithm}_{bag}_error.txt', 'r') as file:
            content = file.readlines()
            for line in content:
                row = line.split()
                time.append(float(row[0]))
                error.append(float(row[1]))

        mean_data_array = np.asarray(error)*1000

        average_smoothed_error = pd.Series(list(mean_data_array)).rolling(window_size).mean().tail(-window_size)
        std_smoothed_error = pd.Series(list(mean_data_array)).rolling(window_size).std().tail(-window_size)

        # plt.plot(mean_data_array, label="Average Frame Error", alpha=0.1)
        ax.plot(average_smoothed_error, label=f'{algorithms_plot[algorithm]}', alpha=1.0, color=colors[i])
        ax.fill_between(average_smoothed_error.index, (average_smoothed_error - std_smoothed_error), (average_smoothed_error + std_smoothed_error), alpha=0.2, color=colors[i])

labelLines(ax.get_lines(), align=False, zorder=2.5)
plt.title('Stationary Rope')
plt.xlabel('Frame Index')
plt.ylabel('Frame Error (mm)')
# plt.legend(framealpha=1, frameon=True)
plt.savefig(f'{ROOT_DIR}/data/frame_error_eval_{bag}.png')
# plt.cla()