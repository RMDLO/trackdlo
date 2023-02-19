import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from labellines import labelLine, labelLines
# , 'gltp', 'cdcpd'
algorithms = ['trackdlo', 'gltp', 'cdcpd']
algorithms_plot = {'trackdlo': 'TrackDLO',
                    'gltp': 'GLTP',
                    'cdcpd': 'CDCPD'}
pct_occlusion = [0,25,50,75]
colors = ['g','b','c','r']
window_size = 100
ROOT_DIR = os.path.abspath(os.curdir)
idx=0
ax = plt.gca()
for occ in pct_occlusion:
    for i, algorithm in enumerate(algorithms):
        for trial in range(1,11):
            files = os.listdir(f'{ROOT_DIR}/src/trackdlo/data/output/{algorithm}')
            data_list = []
            for file in files:
                f = open(f'{ROOT_DIR}/src/trackdlo/data/output/{algorithm}/frame_error_eval_{algorithm}_{trial}_{occ}.json')
                data = json.load(f)
                data_list.append(data['error'])

        mean_data_array = np.asarray(data_list).mean(axis=0)*1000
        std_data_array = np.asarray(data_list).std(axis=0)*1000

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
    plt.savefig(f'{ROOT_DIR}/src/trackdlo/data/output/frame_error_eval_{occ}.png')
    plt.cla()
