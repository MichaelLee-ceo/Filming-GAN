import os
import torch
import torch.optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('[INFO] Creating dir:', path)


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


def plot_traj(real_traj, fake_traj, real_traj_len, pic_path, image_path='trajectory'):

    image_path = pic_path + image_path

    real_traj = real_traj.detach().numpy()
    fake_traj = fake_traj.detach().numpy()
    fake_traj = np.around(fake_traj, decimals=4)

    real_x, real_y, real_z = real_traj[:, 0], real_traj[:, 1], real_traj[:, 2]
    fake_x, fake_y, fake_z = fake_traj[:, 0], fake_traj[:, 1], fake_traj[:, 2]

    # print('\n### Real_traj', real_traj)
    # print('### Fake_traj', fake_traj)
    print('### Difference:', np.sum(real_traj - fake_traj, axis=0))

    fig = plt.figure(figsize=plt.figaspect(0.4))

    # ax = plt.axes(projection='3d')
    # ax.scatter3D(0, 0, 0, c='b')

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(real_x[:real_traj_len], real_y[:real_traj_len], real_z[:real_traj_len], c='dodgerblue')
    ax.plot(real_x[real_traj_len - 1:], real_y[real_traj_len - 1:], real_z[real_traj_len - 1:], c='orange')
    plt.title("Real_traj")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # ax.set_zlim3d(min(real_z), max(np.abs(real_x)))

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot(fake_x[:real_traj_len], fake_y[:real_traj_len], fake_z[:real_traj_len], c='dodgerblue')
    ax.plot(fake_x[real_traj_len - 1:], fake_y[real_traj_len - 1:], fake_z[real_traj_len - 1:], c='orange')
    plt.title("Fake_traj")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # ax.set_zlim3d(min(z), max(np.abs(x)))
    # ax.set_title('Trajectory of video ' + str(i))
    plt.savefig(image_path + '.png', dpi=200)
    print('[INFO] plt.save ' + image_path + '.png')

    # plt.show()
    plt.close()


def check_accuracy(args, loader, model):
    metrics = {}
    traj_l2_losses_abs, traj_l2_losses_rel = ([],) * 2
    disp_error, f_disp_error = ([],) * 2
    total_traj = 0
    loss_mask_sum = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = model(obs_traj_rel)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            traj_l2_loss_abs, traj_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )

            ade = cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped)
            fde = cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped)

            traj_l2_losses_abs.append(traj_l2_loss_abs.item())
            traj_l2_losses_rel.append(traj_l2_loss_rel.item())

            disp_error.append(ade.item())
            f_disp_error.append(fde.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            # if limit and total_traj >= args.num_samples_check:
            #     break

    metrics['traj_l2_loss_abs'] = sum(traj_l2_losses_abs) / loss_mask_sum
    metrics['traj_l2_loss_rel'] = sum(traj_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj

    # wandb.log({'ADE': metrics['ade']})
    # wandb.log({'FDE': metrics['fde']})

    model.train()
    return metrics

def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, loss_mask):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    # ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    # ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade


def cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    # fde_l = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], linear_ped)
    # fde_nl = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped)
    return fde


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))

    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)