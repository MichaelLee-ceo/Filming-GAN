import argparse
import os
import sys
import numpy as np
from trajectories import read_file


all_files = os.listdir('./dataset/opensfm/val')
all_files = [os.path.join('./dataset/opensfm/val', _path) for _path in all_files]

rec_max, rec_min = [], []
for idx, file in enumerate(all_files):
    # print('### Reading file', file)
    data = read_file(file, '\t')            # [ frame_num, peds, x, y, z ]

    current_data = data[:, 2:]
    norm_data = (data[:, 2:] - np.mean(data[:, 2:], axis=0)) / np.std(data[:, 2:], axis=0)

    avg_value = np.sum(norm_data, axis=0)
    max_value, min_value = np.max(norm_data, axis=0), np.min(norm_data, axis=0)

    # print('current\n', current_data)
    # print('norm\n', norm_data)
    rec_max.append(max_value)
    rec_min.append(min_value)

    print('max', max_value, 'min', min_value)
    # print('avg', avg_value.astype('int32'))

print('\n', np.max(rec_max, axis=0), np.min(rec_min, axis=0))
