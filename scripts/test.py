import argparse
import os
import sys
import numpy as np
from fgan.data.trajectories import read_file


all_files = os.listdir('./datasets/opensfm/train')
all_files = [os.path.join('./datasets/opensfm/train', _path) for _path in all_files]

for file in all_files:
    # print('### Reading file', file)
    data = read_file(file, '\t')            # [ frame_num, peds, x, y, z ]

    current_data = data[:, 2:]

    avg_value = sum(current_data) / len(current_data)
    max_value, min_value = np.max(current_data, axis=0), np.min(current_data, axis=0)

    print('\navg', current_data)

    norm_data = (current_data - np.mean(current_data, axis=0)) / np.std(current_data, axis=0)
    print('norm', norm_data)
    input()