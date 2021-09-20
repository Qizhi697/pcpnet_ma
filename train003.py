from __future__ import print_function
import argparse
import os
import sys
import random
import math
import shutil
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from tensorboardX import SummaryWriter  # https://github.com/lanpa/tensorboard-pytorch
import utils
from dataset_our import PointcloudPatchDataset, RandomPointcloudPatchSampler, \
    SequentialShapeRandomPointcloudPatchSampler
from pcpnet003 import PCPNet, MSPCPNet
import numpy as np
import cv2
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--name', type=str, default='my_single_scale_normal', help='training run name')
    parser.add_argument('--desc', type=str, default='My training run for single-scale normal estimation.',
                        help='description')
    parser.add_argument('--indir', type=str, default='./pclouds', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='./models', help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='./logs', help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainingset_whitenoise1.txt', help='training set file name')
    parser.add_argument('--testset', type=str, default='validationset_whitenoise1.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int, default='5', help='save model each n epochs')
    parser.add_argument('--refine', type=str, default='',
                        help='refine model at this path')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+',
                        help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                                                                          'point: center point\n'
                                                                          'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0,
                        help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000,
                        help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100,
                        help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random',
                        help='order in which the training patches are presented:\n'
                             'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                             'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False,
                        help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--use_pca', type=int, default=False,
                        help='Give both inputs and ground truth in local PCA coordinate frame')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean', help='Normal loss type:\n'
                                                                                'ms_euclidean: mean square euclidean distance\n'
                                                                                'ms_oneminuscos: mean square 1-cos(angle error)')

    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals'],
                        help='outputs of the network, a list with elements of:\n'
                             'unoriented_normals: unoriented (flip-invariant) point normals\n'
                             'oriented_normals: oriented point normals\n'
                             'max_curvature: maximum curvature\n'
                             'min_curvature: mininum curvature')
    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1,
                        help='use n-tuples of points as input instead of single points')
    parser.add_argument('--points_per_patch', type=int, default=500, help='max. number of points per patch')

    return parser.parse_args()


def train_pcpnet(opt):
    # opt.gpu_idx = "1,2,3"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 这里的赋值必须是字符串，list会报错
    # device_ids = range(torch.cuda.device_count())
    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)

    # colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_stn003_params.pth' % (opt.name))
    model_filename = os.path.join(opt.outdir, '%s_stn003_model.pth' % (opt.name))
    desc_filename = os.path.join(opt.outdir, '%s_stn003_description.txt' % (opt.name))

    if os.path.exists(log_dirname) or os.path.exists(model_filename):
        # response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (opt.name))
        response = 'y'
        if response == 'y':
            if os.path.exists(log_dirname):
                shutil.rmtree(os.path.join(opt.logdir, opt.name))
        else:
            sys.exit()

    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0
    for o in opt.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')

            output_target_ind.append(target_features.index('normal'))
            output_pred_ind.append(pred_dim)
            output_loss_weight.append(1.0)
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            if o == 'max_curvature':
                output_loss_weight.append(0.7)
            else:
                output_loss_weight.append(0.3)
            pred_dim += 1
        else:
            raise ValueError('Unknown output: %s' % (o))

    if pred_dim <= 0:
        raise ValueError('Prediction is empty for the given outputs.')

    # create model
    if len(opt.patch_radius) == 1:
        pcpnet = PCPNet(
            num_points=opt.points_per_patch,
            output_dim=pred_dim,
            use_point_stn=opt.use_point_stn,
            use_feat_stn=opt.use_feat_stn,
            sym_op=opt.sym_op,
            point_tuple=opt.point_tuple)
    else:
        pcpnet = MSPCPNet(
            num_scales=len(opt.patch_radius),
            num_points=opt.points_per_patch,
            output_dim=pred_dim,
            use_point_stn=opt.use_point_stn,
            use_feat_stn=opt.use_feat_stn,
            sym_op=opt.sym_op,
            point_tuple=opt.point_tuple)

    if opt.refine != '':
        pcpnet.load_state_dict(torch.load(opt.refine))

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # create train and test dataset loaders
    train_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)
    if opt.training_order == 'random':
        train_datasampler = RandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers),
        drop_last=True)
    M = 33
    vec = np.arange(M)
    col, row = np.meshgrid(vec, vec)
    grid_index = np.stack((row.reshape(-1), col.reshape(-1)), axis=1) + 1
    grid_coord = grid_index / M - 1 / (2 * M)
    grid_coord = torch.from_numpy(grid_coord).cuda()
    grid_norm = torch.as_tensor(torch.zeros(opt.batchSize, grid_coord.shape[0], 3), dtype=torch.float32).cuda()
    index = []
    index_null = []
    for i, coord in enumerate(grid_coord):
        nx, ny = 2 * coord[0] - 1, 2 * coord[1] - 1
        if nx ** 2 + ny ** 2 <= 1:
            nz = torch.sqrt(1 - (nx ** 2 + ny ** 2))
            grid_norm[:, i, :] = torch.Tensor([nx, ny, nz])
            index.append(i)
        else:
            grid_norm[:, i, :] = torch.Tensor([100, 100, 100])
            index_null.append(i)
    index_null = torch.tensor(index_null)
    weight_null = torch.ones(M ** 2, opt.points_per_patch).cuda()
    weight_null[index_null, :] = 0
    index_del = torch.ones(opt.batchSize, M ** 2).cuda()
    index_del[:, index_null] = 0
    index_del = index_del.reshape(opt.batchSize, M, M)
    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.testset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)
    if opt.training_order == 'random':
        test_datasampler = RandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        test_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers),
        drop_last=True)

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names
    opt.test_shapes = test_dataset.shape_names

    print('training set: %d patches (in %d batches) - test set: %d patches (in %d batches)' %
          (len(train_datasampler), len(train_dataloader), len(test_datasampler), len(test_dataloader)))

    try:
        os.makedirs(opt.outdir)
    except OSError:
        pass

    train_writer = SummaryWriter(os.path.join(log_dirname, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dirname, 'test'))

    optimizer = optim.SGD(pcpnet.parameters(), lr=opt.lr, momentum=opt.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[],
                                         gamma=0.1)  # milestones in number of optimizer iterations
    # pcpnet.to(device)

    pcpnet = pcpnet.cuda()

    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)

    # save parameters
    torch.save(opt, params_filename)

    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)

    for epoch in range(opt.nepoch):
        train_batchind = -1
        train_fraction_done = 0.0
        train_enum = enumerate(train_dataloader, 0)

        test_batchind = -1
        test_fraction_done = 0.0
        test_enum = enumerate(test_dataloader, 0)

        for train_batchind, data in train_enum:

            # update learning rate
            scheduler.step(epoch * train_num_batch + train_batchind)

            # set to training mode
            pcpnet.train()

            # get trainingset batch and upload to GPU
            points = data[0]
            points = points.transpose(2, 1)
            points = points.cuda()
            targetc = data[1].cuda()

            optimizer.zero_grad()
            pred_map, _, target, trans, _, _ = pcpnet(grid_norm, weight_null, points, targetc, M)

            # 转target
            normal_map = norm_to_grid(target, opt.batchSize, M).cuda()
            normal_map = normal_map * index_del
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_map, normal_map)

            loss.backward()
            # parameter optimization step
            optimizer.step()

            train_fraction_done = (train_batchind + 1) / train_num_batch

            # print info and update log file
            print('[%s %d: %d/%d] %s loss: %f' % (
                opt.name, epoch, train_batchind, train_num_batch - 1, green('train'), loss.item()))
            train_writer.add_scalar('loss', loss.item(),
                                    (epoch + train_fraction_done) * train_num_batch * opt.batchSize)
            # while 0:
            while test_fraction_done <= train_fraction_done and test_batchind + 1 < test_num_batch:
                # set to evaluation mode
                pcpnet.eval()

                test_batchind, data = next(test_enum)

                # get testset batch and upload to GPU
                points = data[0]
                points = points.transpose(2, 1)
                points = points.cuda()
                target = data[1].cuda()
                # forward pass
                with torch.no_grad():
                    # pred_map, trans, _, _ = pcpnet(points, weight, M)
                    # pred_map, _, _, trans, _, _ = pcpnet(grid_norm, weight_null, points, None, M)
                    pred_map, _, _, trans, _, _ = pcpnet(grid_norm, weight_null, points, None, M)
                key_points = torch.sigmoid(pred_map)
                key_points = key_points * index_del
                pred = reverse_mapping(key_points, grid_norm[0, :, :])

                # tt = (dist < 0.01).type(torch.uint8)
                # num = tt.sum(axis=-1)
                # num[:, index_null] = 0
                # point_num = num.reshape(opt.batchSize, M, M)
                # point_num = point_num.cpu().numpy()
                # uu=35
                # t1 = key_points[uu, :, :].detach().cpu().numpy()
                # t2 = normal_map[uu, :, :].detach().cpu().numpy()
                # np.savetxt('points.txt',data[0][uu,:,:].numpy())
                # np.savetxt('normal_map.txt', np.flipud(t2))
                # np.savetxt('pred_map.txt', np.flipud(t1))
                # np.savetxt('pointnum.txt',np.flipud(point_num[uu,:,:]),fmt='%s')
                loss_norm = compute_loss(
                    pred=pred, target=target,
                    outputs=opt.outputs,
                    output_pred_ind=output_pred_ind,
                    output_target_ind=output_target_ind,
                    output_loss_weight=output_loss_weight,
                    patch_rot=trans if opt.use_point_stn else None,
                    normal_loss=opt.normal_loss)
                test_fraction_done = (test_batchind + 1) / test_num_batch

                # print info and update log file
                print('[%s %d: %d/%d] %s loss: %f' % (
                    opt.name, epoch, train_batchind, train_num_batch - 1, blue('test'), loss_norm.item()))
                test_writer.add_scalar('loss', loss.item(),
                                       (epoch + test_fraction_done) * train_num_batch * opt.batchSize)
        if epoch % opt.saveinterval == 0 or epoch == opt.nepoch - 1:
            torch.save(pcpnet.state_dict(), model_filename)


def norm_to_grid(norm, B, M):
    grid = torch.zeros(B, M, M)
    x, y = (norm[:, 0] + 1) * M / 2, (norm[:, 1] + 1) * M / 2
    ind_x, ind_y = x.floor().type(torch.long), y.floor().type(torch.long)
    ind_x[torch.where(ind_x >= M)] = M - 1
    ind_y[torch.where(ind_y >= M)] = M - 1
    grid[torch.arange(0, B), ind_x, ind_y] = 1
    grid = grid.numpy()
    for i in torch.arange(0, B):
        grid[i, ...] = cv2.GaussianBlur(grid[i, ...], (3, 3), 0)
        grid[i, ...] = (grid[i, ...] / np.max(grid[i, ...]))
    return torch.from_numpy(grid)


def reverse_mapping(pred_map, norm):
    B = pred_map.shape[0]
    index = []
    for i in torch.arange(0, B):
        ind = torch.argmax(pred_map[i, :, :])
        index.append(ind)
    pre = norm[index, :]
    return pre


def compute_loss(pred, target, outputs, output_pred_ind, output_target_ind, output_loss_weight, patch_rot, normal_loss):
    loss = 0

    for oi, o in enumerate(outputs):
        if o == 'unoriented_normals' or o == 'oriented_normals':
            # o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi] + 3]
            # o_target = target[output_target_ind[oi]]
            o_pred = pred
            o_target = target

            if patch_rot is not None:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                o_pred = torch.bmm(o_pred.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)

            if o == 'unoriented_normals':
                if normal_loss == 'ms_euclidean':
                    loss += torch.min((o_pred - o_target).pow(2).sum(1), (o_pred + o_target).pow(2).sum(1)).mean() * \
                            output_loss_weight[oi]
                elif normal_loss == 'ms_oneminuscos':
                    loss += (1 - torch.abs(utils.cos_angle(o_pred, o_target))).pow(2).mean() * output_loss_weight[oi]
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss))
            elif o == 'oriented_normals':
                if normal_loss == 'ms_euclidean':
                    loss += (o_pred - o_target).pow(2).sum(1).mean() * output_loss_weight[oi]
                elif normal_loss == 'ms_oneminuscos':
                    loss += (1 - utils.cos_angle(o_pred, o_target)).pow(2).mean() * output_loss_weight[oi]
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss))
            else:
                raise ValueError('Unsupported output type: %s' % (o))

        elif o == 'max_curvature' or o == 'min_curvature':
            o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi] + 1]
            o_target = target[output_target_ind[oi]]

            # Rectified mse loss: mean square of (pred - gt) / max(1, |gt|)
            normalized_diff = (o_pred - o_target) / torch.clamp(torch.abs(o_target), min=1)
            loss += normalized_diff.pow(2).mean() * output_loss_weight[oi]

        else:
            raise ValueError('Unsupported output type: %s' % (o))
    return loss


if __name__ == '__main__':
    train_opt = parse_arguments()
    train_pcpnet(train_opt)
