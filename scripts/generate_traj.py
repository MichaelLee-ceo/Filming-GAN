import os
import argparse
import torch
import torch.optim
import numpy as np
from fgan.data.loader import data_loader
from fgan.utils import mkdir, get_dset_path, relative_to_abs, plot_traj
from fgan.utils import int_tuple, bool_flag
from fgan.models import TrajectoryGenerator

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
# Model Options
parser.add_argument('--embedding_dim', default=16, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=128, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--model_path', default=os.path.join(os.getcwd(), 'models'), type=str)
# Generator Options
parser.add_argument('--encoder_h_dim_g', default=32, type=int)
parser.add_argument('--decoder_h_dim_g', default=32, type=int)
parser.add_argument('--noise_dim', default=(8,), type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='global')
# Pooling Options
parser.add_argument('--pooling_type', default=None)
parser.add_argument('--pool_every_timestep', default=0, type=bool_flag)
# Pool Net Option
parser.add_argument('--bottleneck_dim', default=32, type=int)
# Social Pooling Options
parser.add_argument('--neighborhood_size', default=0.0, type=float)
parser.add_argument('--grid_size', default=0, type=int)
# Generation options
parser.add_argument('--best_epoch', default=236, type=int)
parser.add_argument('--gen_len', default=16, type=int)
args = parser.parse_args()


val_path = get_dset_path(args.dataset_name, 'val')
val_dset, val_loader = data_loader(args, val_path)
print('### len(val_dset):', len(val_dset)) 
print('### len(val_loader):', len(val_loader)) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\n[INFO] Using', device, 'for generation.')


best_epoch = args.best_epoch
gen_len = args.gen_len

generator = TrajectoryGenerator(
                obs_len=args.obs_len,
                pred_len=args.pred_len,
                embedding_dim=args.embedding_dim,
                encoder_h_dim=args.encoder_h_dim_g,
                decoder_h_dim=args.decoder_h_dim_g,
                mlp_dim=args.mlp_dim,
                num_layers=args.num_layers,
                noise_dim=args.noise_dim,
                noise_type=args.noise_type,
                noise_mix_type=args.noise_mix_type,
                pooling_type=args.pooling_type,
                pool_every_timestep=args.pool_every_timestep,
                dropout=args.dropout,
                bottleneck_dim=args.bottleneck_dim,
                neighborhood_size=args.neighborhood_size,
                grid_size=args.grid_size,
                batch_norm=args.batch_norm
            )
generator.type(torch.cuda.FloatTensor).eval()
generator.to(device)

# restoring generator from previous checkpoint
restore_path = os.path.join(args.model_path, 'model_%d.pt' % best_epoch)
checkpoint = torch.load(restore_path)
generator.load_state_dict(checkpoint['g_best_state'])
generator.decoder.seq_len = gen_len                                        # set model's predict_length to gen_legnth
print('Restoring from checkpoint {}'.format(restore_path))

gen_path = './(sgan)gen_result/' + 'model_' + str(args.best_epoch) + '/'
mkdir(gen_path)

gen_count = 0
for batch in val_loader:
    if np.random.randint(1, 10) % 3 == 0:               # random selections from val_dataset for generating trajectories
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch

        gen_traj_rel = generator(obs_traj_rel, obs_traj_rel, seq_start_end)
        gen_traj = relative_to_abs(gen_traj_rel, obs_traj[-1])

        traj_gen = torch.cat([obs_traj, gen_traj], dim=0)
        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)

        for bs in range(args.batch_size):
            gen_count += 1
            plot_traj(traj_real[:, bs].cpu(), traj_gen[:, bs].cpu(), args.obs_len, gen_path, 'gen_batch_' + str(gen_count), overlapping=True, to_csv=True)