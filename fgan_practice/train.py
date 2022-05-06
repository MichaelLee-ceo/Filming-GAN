import os
import sys
import logging
import torch
import torch.optim
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from loader import data_loader
import argparse
from collections import defaultdict
import wandb
from utils import mkdir, get_dset_path, relative_to_abs, plot_traj, check_accuracy
from model_zoo.seq2seq import Seq2Seq
from model_zoo.single_lstm import SingleLSTM


FORMAT = '%(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.manual_seed(10)

parser = argparse.ArgumentParser()
# Dataset options
parser.add_argument('--dataset_name', default='opensfm', type=str)
parser.add_argument('--delim', default='tab')
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--model_path', default=os.path.join(os.getcwd(), 'models'), type=str)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--pic_path', default='opensfm_test', type=str)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--drop_out', default=0, type=int)

# Model options
parser.add_argument('--embedding_dim', default=0, type=int)
parser.add_argument('--hidden_dim', default=32, type=int)
parser.add_argument('--num_layers', default=1, type=int)

args = parser.parse_args()

datasets = ['opensfm']
for name in datasets:
    train_path = get_dset_path(name, 'train')
    train_dset, train_loader = data_loader(args, train_path)

    val_path = get_dset_path(name, 'val')
    val_dset, val_loader = data_loader(args, val_path)

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
            name= "dataset(" + name + "), lr(" + str(args.lr) +  "), embedding_dim(" + str(args.embedding_dim) + "), dropout(" + str(args.drop_out) + "), hidden_dim(" + str(args.hidden_dim) + ")",
            config=args,
            reinit=True
        )

    model = Seq2Seq(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, pred_len=pred_len, drop_out=drop_out)
    model.type(torch.cuda.FloatTensor).train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    checkpoint = {
        'best_model_state': None,
        'metrics_train': defaultdict(list),
        'metrics_val': defaultdict(list),
        'best_epoch':0,
    }

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for idx, batch in enumerate(train_loader):
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, non_linear_ped, loss_mask, seq_start_end) = batch

            pred_traj_rel_fake = model(obs_traj_rel)
            # pred_traj_fake = relative_to_abs(pred_traj_rel_fake, obs_traj[-1])

            loss = criterion(pred_traj_rel_fake, pred_traj_rel_gt)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        wandb.log({'loss': total_loss / len(train_loader)})

        logger.info('\nEpoch: ' + str(epoch) + '/' + str(epochs))
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