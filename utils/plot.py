import json
import matplotlib.pyplot as plt
import os

algorithm = 'gltp'
ROOT_DIR = os.path.abspath(os.curdir)
num_files = len(os.listdir(f'{ROOT_DIR}/data/output/{algorithm}'))

for i in range(num_files):
    path = f'{ROOT_DIR}/data/output/{algorithm}/frame_error_eval_{algorithm}_{i+1}.json'
    f = open(path)
    data = json.load(f)
    plt.plot(data['data'])

plt.savefig(f'{ROOT_DIR}/data/output/frame_error_eval_{algorithm}.png')
