import h5py
import torch
from pykitti.utils import read_calib_file
from torch.utils.data import Dataset
import os, os.path
import numpy as np
import random
import pickle
import open3d as o3d
import open3d.core as o3c
import numpy as np

import utils.rotation_conversion as RT


def get_velo(idx, dir, sequence, jitter=False, remove_random_angle=-1, without_ground=False):
    if without_ground:
        velo_path = os.path.join(dir, 'sequences', f'{int(sequence):02d}','velodyne_no_ground_patchwork', f'{idx:06d}.npy')
        # velo_path = os.path.join(dir, 'sequences', f'{int(sequence):02d}','genarative_velodyne', f'{idx:06d}.npy')
        with h5py.File(velo_path, 'r') as hf:
            # scan = hf['PC'][:]
            scan = hf['PC'][:].astype('float32')
    else:
        velo_path = os.path.join(dir, 'sequences', f'{int(sequence):02d}', 'velodyne', f'{idx:06d}.bin')
        scan = np.fromfile(velo_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    if jitter:
        noise = 0.01 * np.random.randn(scan.shape[0], scan.shape[1]).astype(np.float32)
        noise = np.clip(noise, -0.05, 0.05)
        scan = scan + noise

    if remove_random_angle > 0:
        azi = np.arctan2(scan[..., 1], scan[..., 0])
        cols = 2084 * (np.pi - azi) / (2 * np.pi)
        cols = np.minimum(cols, 2084 - 1)
        cols = np.int32(cols)
        start_idx = np.random.randint(0, 2084)
        end_idx = start_idx + (remove_random_angle / (360.0/2084))
        end_idx = int(end_idx % 2084)
        remove_idxs = cols > start_idx
        remove_idxs = remove_idxs & (cols < end_idx)
        scan = scan[np.logical_not(remove_idxs)]

    return scan


class KITTILoader3DPoses(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, poses, train=True, loop_file='loop_GT',
                 jitter=False, remove_random_angle=-1, without_ground=False):
        """

        :param dir: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: semantic-KITTI ground truth poses file
        """

        self.dir = dir
        self.sequence = sequence
        self.jitter = jitter
        self.remove_random_angle = remove_random_angle
        self.without_ground = without_ground
        data = read_calib_file(os.path.join(dir, 'sequences', sequence, 'calib.txt'))
        cam0_to_velo = np.reshape(data['Tr'], (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)
        poses2 = []
        with open(poses, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                pose = torch.zeros((4, 4), dtype=torch.float64)
                pose[0, 0:4] = torch.tensor(x[0:4])
                pose[1, 0:4] = torch.tensor(x[4:8])
                pose[2, 0:4] = torch.tensor(x[8:12])
                pose[3, 3] = 1.0
                pose = cam0_to_velo.inverse() @ (pose @ cam0_to_velo)
                poses2.append(pose.float().numpy())
        self.poses = poses2
        self.train = train

        gt_file = os.path.join(dir, 'sequences', sequence, f'{loop_file}.pickle')
        with open(gt_file, 'rb') as f:
            self.loop_gt = pickle.load(f)
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):

        anchor_pcd = torch.from_numpy(get_velo(idx, self.dir, self.sequence, self.jitter,
                                               self.remove_random_angle, self.without_ground))

        if self.train:
            x = self.poses[idx][0, 3]
            y = self.poses[idx][1, 3]
            z = self.poses[idx][2, 3]

            anchor_pose = torch.tensor([x, y, z])
            possible_match_pose = torch.tensor([0., 0., 0.])

            indices = list(range(len(self.poses)))
            cont = 0
            positive_idx = idx
            negative_idx = idx
            while cont < 2:
                i = random.choice(indices)
                possible_match_pose[0] = self.poses[idx][0, 3]
                possible_match_pose[1] = self.poses[idx][1, 3]
                possible_match_pose[2] = self.poses[idx][2, 3]
                distance = torch.norm(anchor_pose - possible_match_pose)
                if distance <= 4 and idx == positive_idx:
                    positive_idx = i
                    cont += 1
                elif distance > 10 and idx == negative_idx:  # 1.5 < dist < 2.5 -> unknown
                    negative_idx = i
                    cont += 1

            positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.jitter,
                                                     self.remove_random_angle, self.without_ground))
            negative_pcd = torch.from_numpy(get_velo(negative_idx, self.dir, self.sequence, self.jitter,
                                                     self.remove_random_angle, self.without_ground))

            sample = {'anchor': anchor_pcd,
                      'positive': positive_pcd,
                      'negative': negative_pcd}
        else:
            sample = {'anchor': anchor_pcd}

        return sample


class KITTILoader3DDictPairs(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, poses, loop_file='loop_GT', jitter=False, without_ground=False):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        super(KITTILoader3DDictPairs, self).__init__()

        self.jitter = jitter
        self.dir = dir
        self.sequence = int(sequence)
        self.without_ground = without_ground
        data = read_calib_file(os.path.join(dir, 'sequences', sequence, 'calib.txt'))
        cam0_to_velo = np.reshape(data['Tr'], (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)
        poses2 = []
        with open(poses, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                pose = torch.zeros((4, 4), dtype=torch.float64)
                pose[0, 0:4] = torch.tensor(x[0:4])
                pose[1, 0:4] = torch.tensor(x[4:8])
                pose[2, 0:4] = torch.tensor(x[8:12])
                pose[3, 3] = 1.0
                pose = cam0_to_velo.inverse() @ (pose @ cam0_to_velo)
                poses2.append(pose.float().numpy())
        self.poses = poses2
        gt_file = os.path.join(dir, 'sequences', sequence, f'{loop_file}.pickle')
        with open(gt_file, 'rb') as f:
            self.loop_gt = pickle.load(f)
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

    def draw_registration_result(self, source, target, transformation):
        source_temp = source.clone()
        target_temp = target.clone()

        source_temp.transform(transformation)

        # This is patched version for tutorial rendering.
        # Use `draw` function for you application.
        o3d.visualization.draw_geometries(
            [source_temp.to_legacy(),
            target_temp.to_legacy()],
            zoom=0.4459,
            front=[0.9288, -0.2951, -0.2242],
            lookat=[1.6784, 2.0612, 1.4451],
            up=[-0.3402, -0.9189, -0.1996])


    def overap_mask(self, A, B):
        N_A, C = A.shape
        N_B, C = B.shape

        A_color = torch.tensor([1,0,0], dtype=torch.float32)
        A_color = A_color.expand(N_A,3)
        B_color = torch.tensor([0,1,0], dtype=torch.float32)
        B_color = B_color.expand(N_B,3)

        source = o3d.t.geometry.PointCloud()
        source.point.positions = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(A))
        target = o3d.t.geometry.PointCloud()
        target.point.positions = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(B))
        source.point.colors = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(A_color))
        target.point.colors = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(B_color))
        
        max_correspondence_distance = 0.2
        init_source_to_target = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]])

        estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()

        criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)

        registration_icp = o3d.t.pipelines.registration.icp(source, target, max_correspondence_distance,
                                    init_source_to_target, estimation, criteria)
        temp = torch.utils.dlpack.from_dlpack(registration_icp.transformation.to_dlpack()).transpose(0,1)
        #self.draw_registration_result(source, target, registration_icp.transformation)
        return temp

    def __len__(self):
        return len(self.loop_gt)

    def __getitem__(self, idx):
        frame_idx = self.loop_gt[idx]['idx']
        if frame_idx >= len(self.poses):
            print(f"ERRORE: sequence {self.sequence}, frame idx {frame_idx} ")

        anchor_pcd = torch.from_numpy(get_velo(frame_idx, self.dir, self.sequence, self.jitter, without_ground = self.without_ground))

        #Random permute points
        random_permute = torch.randperm(anchor_pcd.shape[0])
        anchor_pcd = anchor_pcd[random_permute]

        anchor_pose = self.poses[frame_idx]
        anchor_transl = torch.tensor(anchor_pose[:3, 3], dtype=torch.float32)

        positive_idx = np.random.choice(self.loop_gt[idx]['positive_idxs'])

        positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.jitter, without_ground = self.without_ground))

        #Random permute points
        random_permute = torch.randperm(positive_pcd.shape[0])
        positive_pcd = positive_pcd[random_permute]


        if positive_idx >= len(self.poses):
            print(f"ERRORE: sequence {self.sequence}, positive idx {positive_idx} ")
        positive_pose = self.poses[positive_idx]
        positive_transl = torch.tensor(positive_pose[:3, 3], dtype=torch.float32)

        #negative_idx = np.random.choice(self.loop_gt[idx]['negative_idxs'])

        #negative_pcd = torch.from_numpy(get_velo(negative_idx, self.dir, self.sequence, self.jitter, self.without_ground))

        #Random permute points
        #random_permute = torch.randperm(negative_pcd.shape[0])
        #negative_pcd = negative_pcd[random_permute]

        #if negative_idx >= len(self.poses):
        #    print(f"ERRORE: sequence {self.sequence}, negative idx {negative_idx} ")
        #negative_pose = self.poses[negative_idx]
        #negative_transl = torch.tensor(negative_pose[:3, 3], dtype=torch.float32)

        r_anch = anchor_pose
        r_pos = positive_pose
        #r_neg = negative_pose
        r_anch = RT.npto_XYZRPY(r_anch)[3:]
        r_pos = RT.npto_XYZRPY(r_pos)[3:]
        #r_neg = RT.npto_XYZRPY(r_neg)[3:]

        anchor_rot_torch = torch.tensor(r_anch.copy(), dtype=torch.float32)
        positive_rot_torch = torch.tensor(r_pos.copy(), dtype=torch.float32)
        #negative_rot_torch = torch.tensor(r_neg.copy(), dtype=torch.float32)
        #self.overap_mask(anchor_pcd[:,:3], positive_pcd[:,:3])
        
        sample = {'anchor': anchor_pcd,
                  'positive': positive_pcd,
                  #'negative': negative_pcd,
                  'sequence': self.sequence,
                  'anchor_pose': anchor_transl,
                  'positive_pose': positive_transl,
                  #'negative_pose': negative_transl,
                  'anchor_rot': anchor_rot_torch,
                  'positive_rot': positive_rot_torch,
                  #'negative_rot': negative_rot_torch,
                  'anchor_idx': frame_idx,
                  'positive_idx': positive_idx
                  #'negative_idx': negative_idx
                  }

        return sample
