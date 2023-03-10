import argparse
import os
from shutil import copy2
from torchinfo import summary
import yaml
import wandb
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import random
from torch.nn.parallel import DistributedDataParallel
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
from datasets.KITTI360Dataset import KITTI3603DDictPairs, KITTI3603DPoses
from datasets.KITTIDataset import KITTILoader3DPoses, KITTILoader3DDictPairs
from loss import TripletLoss, sinkhorn_matches_loss, pose_loss, pose_negative_loss

from models.get_models import get_model
from triple_selector import hardest_negative_selector, random_negative_selector, \
    semihard_negative_selector
from utils.data import datasets_concat_kitti, merge_inputs, datasets_concat_kitti360
from evaluate_kitti import evaluate_model_with_emb
from datetime import datetime

from utils.geometry import get_rt_matrix, mat2xyzrpy
from utils.tools import _pairwise_distance
from pytorch_metric_learning import distances

import torch.distributed as dist
import torch.multiprocessing as mp

torch.backends.cudnn.benchmark = True

EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
    seed = seed + worker_id + epoch * 100
    seed = seed % (2**32 - 1)
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(model, optimizer, sample, loss_fn, exp_cfg, device, mode='pairs'):
    if True:
        model.train()

        optimizer.zero_grad()

        if 'sequence' in sample:
            neg_mask = sample['sequence'].view(1,-1) != sample['sequence'].view(-1, 1)
        else:
            neg_mask = torch.zeros((sample['anchor_pose'].shape[0], sample['anchor_pose'].shape[0]),
                                   dtype=torch.bool)

        pair_dist = _pairwise_distance(sample['anchor_pose'])
        neg_mask = ((pair_dist > exp_cfg['negative_distance']) | neg_mask)
        neg_mask = neg_mask.repeat(2, 2).to(device)

        anchor_transl = sample['anchor_pose'].to(device)
        positive_transl = sample['positive_pose'].to(device)
        negative_transl = sample['negative_pose'].to(device)
        anchor_rot = sample['anchor_rot'].to(device)
        positive_rot = sample['positive_rot'].to(device)
        negative_rot = sample['negative_rot'].to(device)

        anchor_list = []
        positive_list = []

        delta_pose = []
        ramda_pose = []
        for i in range(anchor_transl.shape[0]):
            anchor = sample['anchor'][i].to(device)
            positive = sample['positive'][i].to(device)

            anchor_i = anchor
            positive_i = positive
            anchor_transl_i = anchor_transl[i]
            anchor_rot_i = anchor_rot[i]
            positive_transl_i = positive_transl[i]
            positive_rot_i = positive_rot[i]
            negative_transl_i = negative_transl[i]
            negative_rot_i = negative_rot[i]

            anchor_i_reflectance = anchor_i[:, 3].clone()
            positive_i_reflectance = positive_i[:, 3].clone()
            anchor_i[:, 3] = 1.
            positive_i[:, 3] = 1.

            rt_anchor = get_rt_matrix(anchor_transl_i, anchor_rot_i, rot_parmas='xyz')
            rt_positive = get_rt_matrix(positive_transl_i, positive_rot_i, rot_parmas='xyz')
            rt_negative = get_rt_matrix(negative_transl_i, negative_rot_i, rot_parmas='xyz')

            if exp_cfg['point_cloud_augmentation']:

                rotz = np.random.rand() * 360 - 180
                rotz = rotz * (np.pi / 180.0)

                roty = (np.random.rand() * 6 - 3) * (np.pi / 180.0)
                rotx = (np.random.rand() * 6 - 3) * (np.pi / 180.0)

                T = torch.rand(3)*3. - 1.5
                T[-1] = torch.rand(1)*0.5 - 0.25
                T = T.to(device)

                rt_anch_augm = get_rt_matrix(T, torch.tensor([rotx, roty, rotz]).to(device))
                anchor_i = rt_anch_augm.inverse() @ anchor_i.T
                anchor_i = anchor_i.T
                anchor_i[:, 3] = anchor_i_reflectance.clone()

                rotz = np.random.rand() * 360 - 180
                rotz = rotz * (3.141592 / 180.0)

                roty = (np.random.rand() * 6 - 3) * (np.pi / 180.0)
                rotx = (np.random.rand() * 6 - 3) * (np.pi / 180.0)

                T = torch.rand(3)*3.-1.5
                T[-1] = torch.rand(1)*0.5 - 0.25
                T = T.to(device)

                rt_pos_augm = get_rt_matrix(T, torch.tensor([rotx, roty, rotz]).to(device))
                positive_i = rt_pos_augm.inverse() @ positive_i.T
                positive_i = positive_i.T
                positive_i[:, 3] = positive_i_reflectance.clone()

                rt_anch_concat = rt_anchor @ rt_anch_augm
                rt_pos_concat = rt_positive @ rt_pos_augm
                rt_neg_concat = rt_negative @ rt_pos_augm

                rt_anchor2positive = rt_anch_concat.inverse() @ rt_pos_concat
                rt_anchor2negative = rt_anch_concat.inverse() @ rt_neg_concat
                ext = mat2xyzrpy(rt_anchor2positive)

            else:
                raise NotImplementedError()

            anchor_list.append(model.module.backbone.prepare_input(anchor_i))
            positive_list.append(model.module.backbone.prepare_input(positive_i))
            del anchor_i, positive_i
            delta_pose.append(rt_anchor2positive.unsqueeze(0))
            ramda_pose.append(rt_anchor2negative.unsqueeze(0))

        delta_pose = torch.cat(delta_pose)
        ramda_pose = torch.cat(ramda_pose)

        model_in = KittiDataset.collate_batch(anchor_list + positive_list)
        for key, val in model_in.items():
            if not isinstance(val, np.ndarray):
                continue
            model_in[key] = torch.from_numpy(val).float().to(device)
        
        #print(model_in['points'].shape)
        #print(anchor_list[0]['points'].shape)
        #print(anchor_list[1]['points'].shape)
        #print(positive_list[0]['points'].shape)
        #print(positive_list[1]['points'].shape)
        metric_head = True
        compute_embeddings = True
        compute_transl = True
        compute_rotation = True
        batch_dict = model(model_in, metric_head, compute_embeddings,
                           compute_transl, compute_rotation, mode=mode)

        model_out = batch_dict['out_embedding']

        # Translation loss
        total_loss = 0.

        loss_transl = torch.tensor([0.], device=device)

        if exp_cfg['weight_rot'] > 0.:
            if exp_cfg['sinkhorn_aux_loss']:
                aux_loss = sinkhorn_matches_loss(batch_dict, delta_pose, mode=mode)
            else:
                aux_loss = torch.tensor([0.], device=device)
            #print(batch_dict.keys())
            #print(sample.keys())
            loss_rot = pose_loss(batch_dict, delta_pose, mode=mode)
            loss_neg_rot = pose_negative_loss(batch_dict, ramda_pose, mode=mode)
            if exp_cfg['sinkhorn_type'] == 'rpm':
                inlier_loss = (1 - batch_dict['transport'].sum(dim=1)).mean()
                inlier_loss += (1 - batch_dict['transport'].sum(dim=2)).mean()
                loss_rot += 0.01 * inlier_loss

            total_loss = total_loss + exp_cfg['weight_rot']*(loss_rot + 0.05*aux_loss - 0.05*loss_neg_rot)
        else:
            loss_rot = torch.tensor([0.], device=device)

        if exp_cfg['weight_metric_learning'] > 0.:
            if exp_cfg['norm_embeddings']:
                model_out = model_out / model_out.norm(dim=1, keepdim=True)

            pos_mask = torch.zeros((model_out.shape[0], model_out.shape[0]), device=device)

            batch_size = (model_out.shape[0]//2)
            for i in range(batch_size):
                pos_mask[i, i+batch_size] = 1
                pos_mask[i+batch_size, i] = 1
            #print(pos_mask)
            #print(neg_mask)
            loss_metric_learning = loss_fn(model_out, pos_mask, neg_mask) * exp_cfg['weight_metric_learning']
            total_loss = total_loss + loss_metric_learning

        total_loss.backward()
        optimizer.step()

        return total_loss, loss_rot, loss_transl


def test(model, sample, exp_cfg, device):
    model.eval()

    with torch.no_grad():
        anchor_transl = sample['anchor_pose'].to(device)
        positive_transl = sample['positive_pose'].to(device)
        anchor_rot = sample['anchor_rot'].to(device)
        positive_rot = sample['positive_rot'].to(device)

        anchor_list = []
        positive_list = []
        delta_transl_list = []
        delta_rot_list = []
        delta_pose_list = []
        for i in range(anchor_transl.shape[0]):
            anchor = sample['anchor'][i].to(device)
            positive = sample['positive'][i].to(device)

            anchor_i = anchor
            positive_i = positive

            anchor_transl_i = anchor_transl[i]
            anchor_rot_i = anchor_rot[i]
            positive_transl_i = positive_transl[i]
            positive_rot_i = positive_rot[i]

            rt_anchor = get_rt_matrix(anchor_transl_i, anchor_rot_i, rot_parmas='xyz')
            rt_positive = get_rt_matrix(positive_transl_i, positive_rot_i, rot_parmas='xyz')
            rt_anchor2positive = rt_anchor.inverse() @ rt_positive
            ext = mat2xyzrpy(rt_anchor2positive)
            delta_transl_i = ext[0:3]
            delta_rot_i = ext[3:]
            delta_transl_list.append(delta_transl_i.unsqueeze(0))
            delta_rot_list.append(delta_rot_i.unsqueeze(0))
            delta_pose_list.append(rt_anchor2positive.unsqueeze(0))

            anchor_list.append(model.module.backbone.prepare_input(anchor_i))
            positive_list.append(model.module.backbone.prepare_input(positive_i))
            del anchor_i, positive_i

        delta_rot = torch.cat(delta_rot_list)
        delta_pose_list = torch.cat(delta_pose_list)

        model_in = KittiDataset.collate_batch(anchor_list + positive_list)
        for key, val in model_in.items():
            if not isinstance(val, np.ndarray):
                continue
            model_in[key] = torch.from_numpy(val).float().to(device)

        batch_dict = model(model_in, metric_head=True)
        anchor_out = batch_dict['out_embedding']

        diff_yaws = delta_rot[:, 2] % (2*np.pi)

        transformation = batch_dict['transformation']
        homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
        transformation = torch.cat((transformation, homogeneous), dim=1)
        transformation = transformation.inverse()
        final_yaws = torch.zeros(transformation.shape[0], device=transformation.device,
                                 dtype=transformation.dtype)
        for i in range(transformation.shape[0]):
            final_yaws[i] = mat2xyzrpy(transformation[i])[-1]
        yaw = final_yaws
        transl_comps_error = (transformation[:,:3,3] - delta_pose_list[:,:3,3]).norm(dim=1).mean()

        yaw = yaw % (2*np.pi)
        yaw_error_deg = torch.abs(diff_yaws - yaw)
        yaw_error_deg[yaw_error_deg>np.pi] = 2*np.pi - yaw_error_deg[yaw_error_deg>np.pi]
        yaw_error_deg = yaw_error_deg.mean() * 180 / np.pi

    if exp_cfg['norm_embeddings']:
        anchor_out = anchor_out / anchor_out.norm(dim=1, keepdim=True)
    anchor_out = anchor_out[:anchor_transl.shape[0]]
    return anchor_out, transl_comps_error, yaw_error_deg


def get_database_embs(model, sample, exp_cfg, device):
    model.eval()

    with torch.no_grad():
        anchor_list = []
        for i in range(len(sample['anchor'])):
            anchor = sample['anchor'][i].to(device)

            anchor_i = anchor
            anchor_list.append(model.module.backbone.prepare_input(anchor_i))
            del anchor_i

        model_in = KittiDataset.collate_batch(anchor_list)
        for key, val in model_in.items():
            if not isinstance(val, np.ndarray):
                continue
            model_in[key] = torch.from_numpy(val).float().to(device)

        batch_dict = model(model_in, metric_head=False)
        anchor_out = batch_dict['out_embedding']

    if exp_cfg['norm_embeddings']:
        anchor_out = anchor_out / anchor_out.norm(dim=1, keepdim=True)
    return anchor_out


def main_process(gpu, exp_cfg, common_seed, world_size, args):
    model = get_model(exp_cfg)
    if args.weights is not None:
        print('Loading pre-trained params...')
        saved_params = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(saved_params['state_dict'])

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.train()
    print(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', default='./KITTI',
                        help='dataset directory')
    parser.add_argument('--dataset', default='kitti',
                        help='dataset')
    parser.add_argument('--without_ground', action='store_true', default=False,
                        help='Use preprocessed point clouds with ground plane removed')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Batch size (per gpu). Minimum 2.')
    parser.add_argument('--checkpoints_dest', default='./checkpoints',
                        help='folder where to save checkpoints')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Activate wandb service')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Only use default value: -1')
    parser.add_argument('--gpu_count', type=int, default=-1,
                        help='Only use default value: -1')
    parser.add_argument('--port', type=str, default='8890',
                        help='port to be used for DDP multi-gpu training')
    parser.add_argument('--weights', type=str, default=None,
                        help='Weights to be loaded, use together with --resume'
                             'to resume a previously stopped training')
    parser.add_argument('--resume', action='store_true',
                        help='Add this flag to resume a previously stopped training,'
                             'the --weights argument must be provided.')

    args = parser.parse_args()

    if args.batch_size < 2:
        raise argparse.ArgumentTypeError("The batch size should be at least 2")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    if not args.wandb:
        os.environ['WANDB_MODE'] = 'dryrun'

    with open("wandb_config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    if args.gpu_count == -1:
        args.gpu_count = torch.cuda.device_count()
    if args.gpu == -1:
        mp.spawn(main_process, nprocs=args.gpu_count, args=(cfg['experiment'], 42, args.gpu_count, args,))
    else:
        main_process(args.gpu, cfg['experiment'], 42, args.gpu_count, args)
