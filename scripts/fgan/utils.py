import os
import csv
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import subprocess
from fgan.losses import l2_loss
from fgan.losses import displacement_error, final_displacement_error

torch.manual_seed(10)
np.random.seed(10)


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def find_nan(variable, var_name):
    variable_n = variable.data.cpu().numpy()
    if np.isnan(variable_n).any():
        exit('%s has nan' % var_name)


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def lineno():
    return str(inspect.currentframe().f_back.f_lineno)


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm


@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))


def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(1), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[0].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    return os.path.join(_dir, 'datasets', dset_name, dset_type)


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


def check_accuracy(
    args, loader, generator, discriminator, d_loss_fn, limit=True
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj

    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('[INFO] Creating dir:', path)


def plot_traj(real_traj, fake_traj, real_traj_len, pic_path, img_path='trajectory', overlapping=False, to_csv=False):

    image_path = pic_path + img_path

    real_traj = real_traj.detach().numpy()
    fake_traj = fake_traj.detach().numpy()
    fake_traj = np.around(fake_traj, decimals=4)

    real_x, real_y, real_z = real_traj[:, 0], real_traj[:, 1], real_traj[:, 2]
    fake_x, fake_y, fake_z = fake_traj[:, 0], fake_traj[:, 1], fake_traj[:, 2]

    # print('\n### Real_traj', real_traj)
    # print('### Fake_traj', fake_traj)
    # print('### Difference:', np.sum(real_traj - fake_traj, axis=0))

    # fig = plt.figure(figsize=plt.figaspect(0.4))
    fig = plt.figure()

    # ax = plt.axes(projection='3d')
    # ax.scatter3D(0, 0, 0, c='b')

    if not overlapping:
        # plot figure 1
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot(real_x[:real_traj_len], real_y[:real_traj_len], real_z[:real_traj_len], c='gray')
        ax.plot(real_x[real_traj_len - 1:], real_y[real_traj_len - 1:], real_z[real_traj_len - 1:], c='orange')
        plt.title("Real_traj")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot(fake_x[:real_traj_len], fake_y[:real_traj_len], fake_z[:real_traj_len], c='gray')
        ax.plot(fake_x[real_traj_len - 1:], fake_y[real_traj_len - 1:], fake_z[real_traj_len - 1:], c='fuchsia')
        plt.title("Fake_traj")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    else:
        # plot figure 2
        ax = plt.axes(projection='3d')
        ax.plot(real_x[:real_traj_len], real_y[:real_traj_len], real_z[:real_traj_len], c='gray')
        ax.plot(real_x[real_traj_len - 1:], real_y[real_traj_len - 1:], real_z[real_traj_len - 1:], c='orange')
        ax.plot(fake_x[:real_traj_len], fake_y[:real_traj_len], fake_z[:real_traj_len], c='gray')
        ax.plot(fake_x[real_traj_len - 1:], fake_y[real_traj_len - 1:], fake_z[real_traj_len - 1:], c='fuchsia')
        plt.title("Drone_Trajectory")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    plt.savefig(image_path + '.png', dpi=200)
    print('[INFO] plt.save ' + image_path + '.png')

    # plt.show()
    plt.close()

    
    if to_csv:
        mkdir(pic_path + 'csv/real/')
        real_stack = np.transpose([real_x, real_y, real_z])
        real = torch.tensor(real_stack, dtype=torch.float32)
        with open(pic_path + 'csv/real/' + img_path + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(real.tolist())
        print('[INFO] Saving' + pic_path + 'csv/real/' + img_path + '.csv')

        mkdir(pic_path + 'csv/fake/')
        fake_stack = np.transpose([fake_x, fake_y, fake_z])
        fake = torch.tensor(fake_stack, dtype=torch.float32)
        with open(pic_path + 'csv/fake/' + img_path + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(fake.tolist())
        print('[INFO] Saving' + pic_path + 'csv/fake/' + img_path + '.csv')