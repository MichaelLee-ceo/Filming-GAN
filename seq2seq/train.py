import os
import sys
import time
import logging
import torch
import torch.optim
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from data.loader import data_loader, dataset_loader
import argparse
from collections import defaultdict
import wandb
from utils import mkdir, get_dset_path, relative_to_abs, plot_traj, check_accuracy, l2_loss
from model_zoo.seq2seq import Seq2Seq


FORMAT = '%(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

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
parser.add_argument('--pic_path', default='opensfm_test', type=str)
parser.add_argument('--model_path', default=os.path.join(os.getcwd(), 'models'), type=str)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--drop_out', default=0, type=int)

# Model options
parser.add_argument('--embedding_dim', default=0, type=int)
parser.add_argument('--hidden_dim', default=256, type=int)
parser.add_argument('--num_layers', default=1, type=int)

def main(args):
    datasets = ['opensfm']
    for name in datasets:
        train_path = get_dset_path(name, 'train')
        dataset = dataset_loader(args, train_path)

        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size

        train_dset, val_dset = random_split(dataset, [train_size, val_size])

        train_loader = data_loader(args, train_dset)
        val_loader = data_loader(args, val_dset)

        print('\n### len(train_dset):', len(train_dset))
        print('### len(val_dset):', len(val_dset))
        # print('### len(train_loader)', len(train_loader))

        pic_path = './pic_result/' + name + '_lr' + str(args.lr) + 'dp' + str(args.drop_out) + 'em' + str(args.embedding_dim) + 'hd' + str(args.hidden_dim) + '/'
        mkdir(pic_path)
        mkdir(args.model_path)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\n[INFO] Using', device, 'for training.')

        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        pred_len = args.pred_len
        epochs = args.num_epochs
        lr = args.lr
        drop_out = args.drop_out

        run = wandb.init(
                project="seq2seq",
                name= "L2(" + name + "), lr(" + str(args.lr) +  "), embedding_dim(" + str(args.embedding_dim) + "), dropout(" + str(args.drop_out) + "), hidden_dim(" + str(args.hidden_dim) + ")",
                config=args,
                reinit=True
            )

        model = Seq2Seq(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, pred_len=pred_len, drop_out=drop_out)
        model.type(torch.cuda.FloatTensor).train()
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # criterion = torch.nn.MSELoss()

        checkpoint = {
            'best_model_state': None,
            'metrics_train': defaultdict(list),
            'metrics_val': defaultdict(list),
            'best_epoch':0,
        }

        for epoch in range(1, epochs + 1):
            total_loss = 0
            # loss_mask_sum = 0

            start  = time.time()
            for idx, batch in enumerate(train_loader):
                batch = [tensor.to(device) for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, non_linear_ped, loss_mask, seq_start_end) = batch

                pred_traj_rel_fake = model(obs_traj_rel)
                # pred_traj_fake = relative_to_abs(pred_traj_rel_fake, obs_traj[-1])

                loss = l2_loss(pred_traj_rel_fake, pred_traj_rel_gt, loss_mask[:, args.obs_len:], mode='average')

                # loss = criterion(pred_traj_rel_fake, pred_traj_rel_gt)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # loss_mask_sum += torch.numel(loss_mask.data)

            wandb.log({'loss': total_loss})

            logger.info('\nEpoch: ' + str(epoch) + '/' + str(args.num_epochs) + ', used ' + str(time.time() - start) + ' s')
            metrics_train = check_accuracy(args, train_loader, model)
            for k, v in sorted(metrics_train.items()):
                logger.info('[train] {}: {:.3f}'.format(k, v))
                checkpoint['metrics_train'][k].append(v)
                wandb.log({'train_' + k: v})

            metrics_val = check_accuracy(args, val_loader, model)
            for k, v in sorted(metrics_val.items()):
                logger.info('[val] {}: {:.3f}'.format(k, v))
                checkpoint['metrics_val'][k].append(v)
                wandb.log({'val_' + k: v})

            min_ade = min(checkpoint['metrics_val']['ade'])

            if metrics_val['ade'] == min_ade:
                logger.info('\nNew low for avg_disp_error')
                checkpoint['best_epoch'] = epoch
                checkpoint['best_model_state'] = model.state_dict()

        checkpoint_path = os.path.join(args.model_path, 'model_%s.pt' % checkpoint['best_epoch'])
        torch.save(checkpoint, checkpoint_path)
        logger.info('Saving checkpoint to {}'.format(checkpoint_path))


        restore_path = checkpoint_path
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        model.load_state_dict(checkpoint['best_model_state'])

        model.eval()
        # testing for training data
        for batch in train_loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch

            pred_traj_fake_rel = model(obs_traj_rel)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)

            for bs in range(args.batch_size):
                plot_traj(traj_real[:, bs].cpu(), traj_fake[:, bs].cpu(), args.obs_len, pic_path, 'train_batch_' + str(bs))

            break

        # testing for validation data
        for batch in val_loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch

            pred_traj_fake_rel = model(obs_traj_rel)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)

            for bs in range(args.batch_size):
                plot_traj(traj_real[:, bs].cpu(), traj_fake[:, bs].cpu(), args.obs_len, pic_path, 'test_batch_' + str(bs))

            break

        run.finish()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)