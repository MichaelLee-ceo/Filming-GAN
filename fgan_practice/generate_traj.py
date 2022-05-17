import os
import torch
import torch.optim
import numpy as np
from loader import data_loader
import argparse
from utils import mkdir, get_dset_path, relative_to_abs, plot_traj
from model_zoo.seq2seq import Seq2Seq

torch.backends.cudnn.benchmark = True
torch.manual_seed(10)
np.random.seed(10)

parser = argparse.ArgumentParser()
# Dataset options
parser.add_argument('--dataset_name', default='opensfm', type=str)
parser.add_argument('--delim', default='tab')
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--model_path', default=os.path.join(os.getcwd(), 'models'), type=str)
# Model options
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--embedding_dim', default=0, type=int)
parser.add_argument('--hidden_dim', default=256, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--drop_out', default=0, type=int)
# Generation options
parser.add_argument('--best_epoch', default=308, type=int)
parser.add_argument('--gen_len', default=16, type=int)
args = parser.parse_args()


val_path = get_dset_path(args.dataset_name, 'val')
val_dset, val_loader = data_loader(args, val_path)
print('### len(val_dset):', len(val_dset)) 
print('### len(val_loader):', len(val_loader)) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\n[INFO] Using', device, 'for generation.')

embedding_dim = args.embedding_dim
hidden_dim = args.hidden_dim
num_layers = args.num_layers
pred_len = args.pred_len
drop_out = args.drop_out
best_epoch = args.best_epoch
gen_len = args.gen_len

model = Seq2Seq(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, pred_len=pred_len, drop_out=drop_out)
model.type(torch.cuda.FloatTensor).eval()
model.to(device)

# restoring model from previous checkpoint
restore_path = os.path.join(args.model_path, 'model_%d.pt' % best_epoch)
checkpoint = torch.load(restore_path)
model.load_state_dict(checkpoint['best_model_state'])
model.decoder.pred_len = gen_len                                        # set model's predict_length to gen_legnth
print('Restoring from checkpoint {}'.format(restore_path))

gen_path = './gen_result/' + 'model_' + str(args.best_epoch) + '/'
mkdir(gen_path)

gen_count = 0
for batch in val_loader:
    if np.random.randint(1, 10) % 3 == 0:               # random selections from val_dataset for generating trajectories
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch

        gen_traj_rel = model(obs_traj_rel)
        gen_traj = relative_to_abs(gen_traj_rel, obs_traj[-1])

        traj_gen = torch.cat([obs_traj, gen_traj], dim=0)
        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)

        for bs in range(args.batch_size):
            gen_count += 1
            plot_traj(traj_real[:, bs].cpu(), traj_gen[:, bs].cpu(), args.obs_len, gen_path, 'gen_batch_' + str(gen_count), overlapping=True, to_csv=True)