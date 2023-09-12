import argparse
import h5py
import open3d as o3d
import os
import torch
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root_folder', default='./KITTI', help='dataset directory')
args = parser.parse_args()
base_dir = args.root_folder
out_dir = 'velodyne_no_ground'

for sequence in ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
                 "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync",
                 "2013_05_28_drive_0006_sync"]:
    if not os.path.exists(os.path.join(base_dir, 'data_3d_raw', sequence, out_dir)):
        os.mkdir(os.path.join(base_dir, 'data_3d_raw', sequence, out_dir))
    f = open(f"./failed_frames_{sequence}.txt", "w")

    file_list = os.listdir(os.path.join(base_dir, 'data_3d_raw', sequence, 'velodyne_points', 'data'))
    for idx in tqdm(range(len(file_list))):
        velo_path = os.path.join(base_dir, 'data_3d_raw', sequence, 'velodyne_points', 'data', f'{idx:010d}.bin')
        save_file = os.path.join(base_dir, 'data_3d_raw', sequence, out_dir, f'{idx:010d}.npy')
        if os.path.exists(save_file):
            continue
        scan = np.fromfile(velo_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scan[:,:3])
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        i = 0
        while np.argmax(plane_model[:-1]) != 2:
            i += 1
            pcd = pcd.select_by_index(inliers, invert=True)
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                                     ransac_n=3,
                                                     num_iterations=10000)
            if i == 5:
                f.write(f'{idx:06d}.bin\n')
                f.flush()
                break
        outliers_index = set(range(scan.shape[0])) - set(inliers)
        outliers_index = list(outliers_index)
        no_ground_scan = scan[outliers_index]

        with h5py.File(save_file, 'w') as hf:
            hf.create_dataset('PC', data=no_ground_scan, compression='lzf', shuffle=True)
    f.close()
