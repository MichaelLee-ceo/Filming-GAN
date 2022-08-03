import os
import random
import torch
import torch.optim
import numpy as np
from torch.utils.data import random_split
from data.loader import dataset_loader, data_loader
import argparse
from utils import mkdir, get_dset_path, relative_to_abs, plot_traj, evaluation
from model_zoo.model import Generator, Discriminator
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
parser.add_argument('--batch_size', default=1, type=int)
# Model Options
parser.add_argument('--embedding_dim', default=0, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--activation', default='relu', type=str)
parser.add_argument('--batch_norm', default=True, type=bool)
# Generator Options
parser.add_argument('--encoder_h_dim_g', default=128, type=int)
parser.add_argument('--decoder_h_dim_g', default=256, type=int)
parser.add_argument('--g_mlp_dim', default=128, type=int)
parser.add_argument('--noise_dim', default=16, type=int)
parser.add_argument('--noise_type', default='gaussian', type=str)
# Generation options
parser.add_argument('--model_path', default=os.path.join(os.getcwd(), 'models'), type=str)
parser.add_argument('--gan_best_epoch', default=458, type=int)

# Seq2Seq options
# parser.add_argument('--embedding_dim', default=0, type=int)
parser.add_argument('--hidden_dim', default=256, type=int)
# parser.add_argument('--num_layers', default=1, type=int)
# parser.add_argument('--drop_out', default=0, type=int)
parser.add_argument('--seq_best_epoch', default=477, type=int)
args = parser.parse_args()


train_path = get_dset_path(args.dataset_name, 'train')
# dataset = dataset_loader(args, train_path)

# train_size = int(len(dataset) * 0.8)
# val_size = len(dataset) - train_size

# train_dset, val_dset = random_split(dataset, [train_size, val_size])
# val_loader = data_loader(args, val_dset)
# print('### len(val_dset):', len(val_dset))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\n[INFO] Using', device, 'for generation.')

'''
  - GAN
'''
generator = Generator(
        embedding_dim = args.embedding_dim,
        encoder_h_dim = args.encoder_h_dim_g,
        decoder_h_dim = args.decoder_h_dim_g,
        num_layers = args.num_layers,
        mlp_dim = args.g_mlp_dim,
        dropout = args.dropout,
        obs_len = args.obs_len,
        pred_len = args.pred_len,
        noise_dim = args.noise_dim,
        noise_type = args.noise_type,
        activation = args.activation,
        batch_norm = args.batch_norm,
    )
generator.type(torch.cuda.FloatTensor).eval()
generator.to(device)
# restoring model from previous checkpoint
gan_restore_path = os.path.join(args.model_path, 'model_%d.pt' % args.gan_best_epoch)
gan_checkpoint = torch.load(gan_restore_path)
generator.load_state_dict(gan_checkpoint['best_g_state'])
print('Restoring from checkpoint {}'.format(gan_restore_path))

'''
 - Seq2Seq
'''
seq2seq = Seq2Seq(embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, pred_len=args.pred_len, drop_out=args.dropout)
seq2seq.type(torch.cuda.FloatTensor).eval()
seq2seq.to(device)
# restoring model from previous checkpoint
seq_restore_path = os.path.join(args.model_path, 'model_%d.pt' % args.seq_best_epoch)
seq_checkpoint = torch.load(seq_restore_path)
seq2seq.load_state_dict(seq_checkpoint['best_model_state'])
print('Restoring from checkpoint {}'.format(seq_restore_path))


# gen_path = './gen_result/' + 'model_' + str(args.best_epoch) + '/' + 'gen_len_' + str(args.pred_len) + '/'
# mkdir(gen_path)

# gen_count = 0
# for batch in val_loader:
#     if np.random.randint(1, 10) % 5 == 0:               # random selections from val_dataset for generating trajectories
#         batch = [tensor.cuda() for tensor in batch]
#         (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch

#         gen_traj_rel = generator(obs_traj_rel)
#         gen_traj = relative_to_abs(gen_traj_rel, obs_traj[-1])

#         traj_gen = torch.cat([obs_traj, gen_traj], dim=0)
#         traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)

#         for bs in range(int(args.batch_size/4)):
#             gen_count += 1
#             plot_traj(traj_real[:, bs].cpu(), traj_gen[:, bs].cpu(), args.obs_len, gen_path, 'gen_batch_' + str(gen_count), overlapping=True, to_csv=True)

val_pred_len = [6, 9, 12]
for p_len in val_pred_len:
    args.pred_len = p_len
    generator.decoder.pred_len = p_len
    seq2seq.decoder.pred_len = p_len
    
    dataset = dataset_loader(args, train_path)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    train_dset, val_dset = random_split(dataset, [train_size, val_size])
    val_loader = data_loader(args, val_dset)


    g_ade, g_fde = evaluation(args, val_loader, generator)
    print('FGAN: Pred Len: {}, ADE: {:.3f}, FDE: {:.3f}'.format(args.pred_len, g_ade, g_fde))
    s_ade, s_fde = evaluation(args, val_loader, seq2seq)
    print('LSTM: Pred Len: {}, ADE: {:.3f}, FDE: {:.3f}'.format(args.pred_len, s_ade, s_fde))
    