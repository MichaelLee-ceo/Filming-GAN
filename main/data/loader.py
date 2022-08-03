import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from .trajectories import TrajectoryDataset, seq_collate

torch.manual_seed(0)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dataset_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)
    return dset

def data_loader(args, dset):
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        num_workers=1,
        collate_fn=seq_collate)
    return loader
