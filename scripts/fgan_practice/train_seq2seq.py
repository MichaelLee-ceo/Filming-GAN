import os
import torch.optim
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils.loader import data_loader
import argparse
import wandb

from seq2seq import Seq2Seq
from single_lstm import SingleLSTM

torch.backends.cudnn.benchmark = True

torch.manual_seed(10)

def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")
    _dir = "/".join(_dir)
    return os.path.join(_dir, "dataset", dset_name, dset_type)

def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def plot_traj(real_traj, fake_traj, real_traj_len, image_path='trajectory'):

    image_path = pic_path + image_path

    real_traj = real_traj.detach().numpy()
    fake_traj = fake_traj.detach().numpy()
    fake_traj = np.around(fake_traj, decimals=4)

    real_x, real_y, real_z = real_traj[:, 0], real_traj[:, 1], real_traj[:, 2]
    fake_x, fake_y, fake_z = fake_traj[:, 0], fake_traj[:, 1], fake_traj[:, 2]

    # print('\n### Real_traj', real_traj)
    # print('### Fake_traj', fake_traj)
    print('\n### Difference:', np.sum(real_traj - fake_traj, axis=0))

    fig = plt.figure(figsize=plt.figaspect(0.4))

    # ax = plt.axes(projection='3d')
    # ax.scatter3D(0, 0, 0, c='b')

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(real_x[:real_traj_len], real_y[:real_traj_len], real_z[:real_traj_len], c='dodgerblue')
    ax.plot(real_x[real_traj_len - 1:], real_y[real_traj_len - 1:], real_z[real_traj_len - 1:], c='orange')
    plt.title("real_traj")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # ax.set_zlim3d(min(real_z), max(np.abs(real_x)))

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot(fake_x[:real_traj_len], fake_y[:real_traj_len], fake_z[:real_traj_len], c='dodgerblue')
    ax.plot(fake_x[real_traj_len - 1:], fake_y[real_traj_len - 1:], fake_z[real_traj_len - 1:], c='orange')
    plt.title("fake_traj")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # ax.set_zlim3d(min(z), max(np.abs(x)))
    # ax.set_title('Trajectory of video ' + str(i))
    plt.savefig(image_path + '.png', dpi=200)
    print('[INFO] plt.save ' + image_path + '.png')

    plt.show()
    plt.close()


parser = argparse.ArgumentParser()
# Dataset options
parser.add_argument('--dataset_name', default='opensfm', type=str)
parser.add_argument('--delim', default='tab')
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--pic_path', default='opensfm_test', type=str)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--drop_out', default=0.2, type=int)

# Model options
parser.add_argument('--embedding_dim', default=16, type=int)
parser.add_argument('--hidden_dim', default=32, type=int)
parser.add_argument('--num_layers', default=1, type=int)

args = parser.parse_args()

datasets = ['create']
for name in datasets:
    train_path = get_dset_path(name, 'train')
    train_dset, train_loader = data_loader(args, train_path)

    val_path = get_dset_path(name, 'val')
    val_dset, val_loader = data_loader(args, val_path)

    # print('\ntrain_dset', train_dset)
    print('### len(train_dset)', len(train_dset))

    # print('train_loader', train_loader)
    print('### len(train_loader)', len(train_loader))

    pic_path = './results/(nb)' + name + '_lr' + str(args.lr) + 'dp' + str(args.drop_out) + 'em' + str(args.embedding_dim) + 'hd' + str(args.hidden_dim) + '/'
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
        print('[INFO] Creating dir:', pic_path)

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
            name= "(nb)dataset(" + name + "), batch_size(" + str(args.batch_size) + "), lr(" + str(args.lr) +  "), num_epochs(" + str(args.num_epochs) + "), dropout(" + str(args.drop_out) + ")",
            config=args,
            reinit=True
          )

    model = Seq2Seq(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, pred_len=pred_len, drop_out=drop_out)
    model.type(torch.cuda.FloatTensor).train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        for idx, batch in enumerate(train_loader):
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, non_linear_ped, loss_mask, seq_start_end) = batch

            pred_traj_rel_fake = model(obs_traj_rel)
            # pred_traj_fake = relative_to_abs(pred_traj_rel_fake, obs_traj[-1])

            loss = criterion(pred_traj_rel_fake, pred_traj_rel_gt)

            wandb.log({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


    model.eval()
    # testing for training data
    for batch in train_loader:
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch

        pred_traj_fake_rel = model(obs_traj_rel)
        print(pred_traj_fake_rel.shape)

        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)

        for bs in range(args.batch_size):
            plot_traj(traj_real[:, bs].cpu(), traj_fake[:, bs].cpu(), args.obs_len, 'train_batch_' + str(bs))

        break

    # testing for validation data
    for batch in val_loader:
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch

        pred_traj_fake_rel = model(obs_traj_rel)
        print(pred_traj_fake_rel.shape)

        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        # print('traj_fake', traj_fake[0], traj_fake[0].shape)

        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        # print('traj_real', traj_real[0], traj_real[0].shape)

        for bs in range(args.batch_size):
            plot_traj(traj_real[:, bs].cpu(), traj_fake[:, bs].cpu(), args.obs_len, 'test_batch_' + str(bs))

        break

    run.finish()