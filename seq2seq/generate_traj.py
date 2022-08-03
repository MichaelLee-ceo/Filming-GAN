import os
import random
import torch
import torch.optim
import numpy as np
from torch.utils.data import random_split
from data.loader import dataset_loader, data_loader
import argparse
from utils import mkdir, get_dset_path, relative_to_abs, plot_traj, evaluate
from model_zoo.seq2seq import Seq2Seq

torch.backends.cudnn.benchmark = True
SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser()
# Dataset options
parser.add_argument('--dataset_name', default='opensfm', type=str)
parser.add_argument('--delim', default='tab')
parser.add_argument('--loader_num_workers', default=1, type=int)
parser.add_argument('--obs_len', default=6, type=int)
parser.add_argument('--pred_len', default=6, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--model_path', default=os.path.join(os.getcwd(), 'models'), type=str)
# Model options
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--embedding_dim', default=0, type=int)
parser.add_argument('--hidden_dim', default=256, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--drop_out', default=0, type=int)
# Generation options
parser.add_argument('--best_epoch', default=442, type=int)
# parser.add_argument('--gen_len', default=24, type=int)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\n[INFO] Using', device, 'for generation.')

embedding_dim = args.embedding_dim
hidden_dim = args.hidden_dim
num_layers = args.num_layers
pred_len = args.pred_len
drop_out = args.drop_out
best_epoch = args.best_epoch
# gen_len = args.gen_len

model = Seq2Seq(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, pred_len=pred_len, drop_out=drop_out)
model.type(torch.cuda.FloatTensor).eval()
model.to(device)

# restoring model from previous checkpoint
restore_path = os.path.join(args.model_path, 'model_%d.pt' % best_epoch)
checkpoint = torch.load(restore_path)
model.load_state_dict(checkpoint['best_model_state'])
# model.decoder.pred_len = gen_len                                        # set model's predict_length to gen_legnth
print('Restoring from checkpoint {}'.format(restore_path))


train_path = get_dset_path(args.dataset_name, 'train')
dataset = dataset_loader(args, train_path)

train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size

train_dset, val_dset = random_split(dataset, [train_size, val_size])
val_loader = data_loader(args, val_dset[0])
print(val_dset[0])

print('### len(val_dset):', len(val_dset)) 
print('### len(val_loader):', len(val_loader)) 
for batch in val_loader:
    print('batch', batch[0])
    input()

gen_path = './gen_result/' + 'model_' + str(args.best_epoch) + '/'
mkdir(gen_path)

# gen_count = 0
# for batch in val_loader:
#     if np.random.randint(1, 10) % 3 == 0:               # random selections from val_dataset for generating trajectories
#         batch = [tensor.cuda() for tensor in batch]
#         (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch

#         gen_traj_rel = model(obs_traj_rel)
#         gen_traj = relative_to_abs(gen_traj_rel, obs_traj[-1])

#         traj_gen = torch.cat([obs_traj, gen_traj], dim=0)
#         traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)

#         for bs in range(args.batch_size):
#             gen_count += 1
#             plot_traj(traj_real[:, bs].cpu(), traj_gen[:, bs].cpu(), args.obs_len, gen_path, 'gen_batch_' + str(gen_count), overlapping=True, to_csv=True)

for batch in val_loader:
    print('batch', batch[0])
    input()
ade, fde = evaluate(args, val_loader, model)
print('Dataset: {}, Pred Len: {}, ADE: {:.3f}, FDE: {:.3f}'.format(args.dataset_name, args.pred_len, ade, fde))