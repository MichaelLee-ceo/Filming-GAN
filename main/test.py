import os
import torch
import torch.optim
import numpy as np
from torch.utils.data import random_split
from data.loader import dataset_loader, data_loader
import argparse
from utils import mkdir, get_dset_path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_traj(real_traj, real_traj_len, pic_path, img_path='trajectory', overlapping=False, to_csv=False):

    image_path = pic_path + img_path
    mkdir(image_path)

    real_traj = real_traj.detach().numpy()
    real_x, real_y, real_z = real_traj[:, 0], real_traj[:, 1], real_traj[:, 2]

    ax = plt.axes(projection='3d')
    ax.view_init(5, 60)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    plt.grid(True)

    for i in range(real_traj_len*2):
        ax.plot(real_x[:i], real_y[:i], real_z[:i], c='fuchsia')
        fig = plt.gcf()
        fig.savefig(image_path + '/' + str(i) + '.png', dpi=200)
        print('[INFO] plt.save ' + image_path + '/' + str(i) + '.png')
    plt.close()

torch.backends.cudnn.benchmark = True
torch.manual_seed(10)
np.random.seed(10)

parser = argparse.ArgumentParser()
# Dataset options
parser.add_argument('--dataset_name', default='opensfm', type=str)
parser.add_argument('--delim', default='tab')
parser.add_argument('--loader_num_workers', default=1, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--batch_size', default=16, type=int)
args = parser.parse_args()

train_path = get_dset_path(args.dataset_name, 'train')
dataset = dataset_loader(args, train_path)

train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size

train_dset, val_dset = random_split(dataset, [train_size, val_size])
train_loader = data_loader(args, train_dset)

mkdir('test')

gen_count=0
first = True
for batch in train_loader:
    if np.random.randint(1, 10) % 2 == 0:               # random selections from val_dataset for generating trajectories
        # if first:
        #     first=False
        #     continue
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch


        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)

        for bs in range(args.batch_size):
            gen_count += 1
            plot_traj(traj_real[:, bs].cpu(), args.obs_len, './test/', 'gen_batch_' + str(gen_count), overlapping=False, to_csv=False)
        break