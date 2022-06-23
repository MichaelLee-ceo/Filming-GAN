import os
import gc
import sys
import argparse
import logging
import wandb
import numpy as np
import torch
from torch.utils.data import random_split
from collections import defaultdict

from data.loader import dataset_loader, data_loader
from losses import gan_g_loss, gan_d_loss, l2_loss
from utils import mkdir, get_dset_path, relative_to_abs, plot_traj, check_accuracy
from model_zoo.model import Generator, Discriminator
from model_zoo.model import get_noise


parser = argparse.ArgumentParser()
FORMAT = '%(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.manual_seed(10)
np.random.seed(10)

# Dataset options
parser.add_argument('--dataset_name', default='opensfm', type=str)
parser.add_argument('--delim', default='tab')
parser.add_argument('--loader_num_workers', default=1, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_epochs', default=500, type=int)

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
parser.add_argument('--clipping_threshold_g', default=1.5, type=float)
parser.add_argument('--g_learning_rate', default=0.001, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Discriminator Options
parser.add_argument('--encoder_h_dim_d', default=256, type=int)
parser.add_argument('--d_mlp_dim', default=128, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)
parser.add_argument('--d_learning_rate', default=0.001, type=float)
parser.add_argument('--d_steps', default=1, type=int)

# Loss Options
parser.add_argument('--l2_loss_weight', default=1, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--model_path', default=os.path.join(os.getcwd(), 'models'), type=str)
parser.add_argument('--pic_path', default='fgan_test', type=str)

def main(args):
    train_path = get_dset_path(args.dataset_name, 'train')
    dataset = dataset_loader(args, train_path)

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    train_dset, val_dset = random_split(dataset, [train_size, val_size])

    train_loader = data_loader(args, train_dset)
    val_loader = data_loader(args, val_dset)

    # val_path = get_dset_path(args.dataset_name, 'val')
    # val_dset, val_loader = data_loader(args, val_path)

    print('\n### len(train_dset):', len(train_dset))
    print('### len(val_dset):', len(val_dset))

    pic_path = './pic_result/(fgan)' + 'ehg' + str(args.encoder_h_dim_g) + 'dhg' + str(args.decoder_h_dim_g) + 'ehd' + str(args.encoder_h_dim_d) + 'gmlp' + str(args.g_mlp_dim) + 'dmlp' + str(args.d_mlp_dim) + 'n' + str(args.noise_dim) + 'l2_' + str(args.l2_loss_weight) + '/'
    mkdir(pic_path)
    mkdir(args.model_path)

    run = wandb.init(
                    project="f-gan",
                    name =  'ehg' + str(args.encoder_h_dim_g) + 'dhg' + str(args.decoder_h_dim_g) + 'ehd' + str(args.encoder_h_dim_d) + 'gmlp' + str(args.g_mlp_dim) + 'dmlp' + str(args.d_mlp_dim) + 'n' + str(args.noise_dim) + 'l2_' + str(args.l2_loss_weight),
                    # name="obs_len(" + str(args.obs_len) + "), pred_len(" + str(args.pred_len) + "), batch_size(" + str(args.batch_size) + "), num_epochs(" + str(args.num_epochs) + "), noise(" + str(args.noise_dim) + ")",
                    config=args,
                    reinit=True
                )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n[INFO] Using', device, 'for training.')

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
    print('[INFO] Generator:')
    print(generator)

    generator.type(torch.cuda.FloatTensor).train()
    generator.to(device)

    discriminator = Discriminator(
            embedding_dim = args.embedding_dim,
            encoder_h_dim = args.encoder_h_dim_d,
            num_layers = args.num_layers,
            mlp_dim = args.d_mlp_dim,
            dropout = args.dropout,
            activation = args.activation,
            batch_norm = args.batch_norm,
        )

    print('[INFO] Discriminator:')
    print(discriminator)

    discriminator.type(torch.cuda.FloatTensor).train()
    generator.to(device)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)

    checkpoint = {
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'best_g_state': None,
            'best_d_state': None,
            'metrics_train': defaultdict(list),
            'metrics_val': defaultdict(list),
            'best_epoch':0,
        }


    # Start training
    for epoch in range(1, args.num_epochs + 1):
        gc.collect()
        g_steps_left = args.g_steps
        d_steps_left = args.d_steps

        for batch in train_loader:
            if g_steps_left > 0:
                g_losses = generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g)
                g_steps_left -= 1
            if d_steps_left > 0:
                d_losses = discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d)
                d_steps_left -= 1

            if g_steps_left > 0 or d_steps_left > 0:
                continue

            g_steps_left = args.g_steps
            d_steps_left = args.d_steps

        # print training losses of generator & discriminator
        for k, v in sorted(g_losses.items()):
            logger.info('  [G] {}: {:.3f}'.format(k, v))
            checkpoint['G_losses'][k].append(v)
            wandb.log({k: v})
        for k, v in sorted(d_losses.items()):
            logger.info('  [D] {}: {:.3f}'.format(k, v))
            checkpoint['D_losses'][k].append(v)
            wandb.log({k: v})
       
        # print training metrics of train_loader & val_loader
        logger.info('\nEpoch: ' + str(epoch) + '/' + str(args.num_epochs))
        metrics_train = check_accuracy(args, train_loader, generator, discriminator, d_loss_fn)
        for k, v in sorted(metrics_train.items()):
            logger.info('  [train] {}: {:.3f}'.format(k, v))
            checkpoint['metrics_train'][k].append(v)
            wandb.log({'train_' + k: v})

        metrics_val = check_accuracy(args, val_loader, generator, discriminator, d_loss_fn)
        for k, v in sorted(metrics_val.items()):
            logger.info('  [val] {}: {:.3f}'.format(k, v))
            checkpoint['metrics_val'][k].append(v)
            wandb.log({'val_' + k: v})

        # saving best model state for the lowest average_displacement_error
        min_ade = min(checkpoint['metrics_val']['ade'])
        if metrics_val['ade'] == min_ade:
            logger.info('[INFO] New low for avg_disp_error')
            checkpoint['best_epoch'] = epoch
            checkpoint['best_g_state'] = generator.state_dict()
            checkpoint['best_d_state'] = discriminator.state_dict()

    checkpoint_path = os.path.join(args.model_path, 'model_%d.pt' % checkpoint['best_epoch'])
    logger.info('Saving checkpoint to {}'.format(checkpoint_path))
    torch.save(checkpoint, checkpoint_path)

    run.finish()

    # restore the generator from the best history state
    restore_from_checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['best_g_state'])

    generator.eval()
    for batch in train_loader:
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch
        
        pred_traj_fake_rel = generator(obs_traj_rel)
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        
        for bs in range(args.batch_size):
            plot_traj(traj_real[:, bs].cpu(), traj_fake[:, bs].cpu(), args.obs_len, pic_path, 'train_batch_' + str(bs))
        break

    for batch in val_loader:
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch
        
        pred_traj_fake_rel = generator(obs_traj_rel)
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        
        for bs in range(args.batch_size):
            plot_traj(traj_real[:, bs].cpu(), traj_fake[:, bs].cpu(), args.obs_len, pic_path, 'test_batch_' + str(bs))
        break



def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = 0
    g_l2_loss_rel = []
    loss_mask = loss_mask[:, args.obs_len:]

    for i in range(args.best_k):
        pred_traj_fake_rel = generator(obs_traj_rel)
        # pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        # appending l2_loss
        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='raw'))

    g_l2_loss_sum_rel = 0
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel
    

    '''
    # mode seeking loss
    noise_shape = (obs_traj_rel.shape[1], args.noise_dim)
    z1 = get_noise(noise_shape, args.noise_type)
    z2 = get_noise(noise_shape, args.noise_type)
    fake_traj_rel_1 = generator(obs_traj_rel, z1)
    fake_traj_rel_2 = generator(obs_traj_rel, z2)

    lz = torch.mean(torch.abs(fake_traj_rel_2 - fake_traj_rel_1)) / torch.mean(torch.abs(z2 - z1))
    loss_lz = 10 * lz
    loss -= loss_lz
    losses['mode_seeking_loss'] = loss_lz
    '''


    # traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake_rel)
    discriminator_loss = g_loss_fn(scores_fake)     # bce with torch.ones_like()

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clipping_threshold_g)
    optimizer_g.step()

    return losses


def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = 0

    pred_traj_fake_rel = generator(obs_traj_rel)
    # pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_real = discriminator(traj_real_rel)
    scores_fake = discriminator(traj_fake_rel)

    loss += d_loss_fn(scores_real, scores_fake)
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clipping_threshold_d)
    optimizer_d.step()

    return losses


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)