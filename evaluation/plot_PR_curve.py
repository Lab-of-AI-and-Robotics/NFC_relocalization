import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import faiss
import numpy as np
import pickle
import os
import scipy.io as sio
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.neighbors import KDTree
from tqdm import tqdm
import torch
import torch.utils.data
from datasets.KITTIDataset import KITTILoader3DPoses


def compute_PR(pair_dist, poses, map_tree_poses, is_distance=True, ignore_last=False):
    real_loop = []
    detected_loop = []
    distances = []
    last = poses.shape[0]
    if ignore_last:
        last = last-1

    for i in tqdm(range(100, last)):
        min_range = max(0, i-50)
        current_pose = poses[i][:3, 3]
        indices = map_tree_poses.query_radius(np.expand_dims(current_pose, 0), 4)
        valid_idxs = list(set(indices[0]) - set(range(min_range, last)))
        if len(valid_idxs) > 0:
            real_loop.append(1)
        else:
            real_loop.append(0)

        if is_distance:
            candidate = pair_dist[i, :i-50].argmin()
            detected_loop.append(-pair_dist[i, candidate])
        else:
            candidate = pair_dist[i, :i-50].argmax()
            detected_loop.append(pair_dist[i, candidate])
        candidate_pose = poses[candidate][:3, 3]
        distances.append(np.linalg.norm(candidate_pose-current_pose))

    distances = np.array(distances)
    detected_loop = -np.array(detected_loop)
    real_loop = np.array(real_loop)
    precision_fn = []
    recall_fn = []
    for thr in np.unique(detected_loop):
        asd = detected_loop<=thr
        asd = asd & real_loop
        asd = asd & (distances <= 4)
        tp = asd.sum()
        fn = (detected_loop<=thr) & (distances > 4) & real_loop
        fn2 = (detected_loop > thr) & real_loop
        fn = fn.sum() + fn2.sum()
        fp = (detected_loop<=thr) & (distances > 4) & (1 - real_loop)
        fp = fp.sum()
        if (tp+fp) > 0:
            precision_fn.append(tp/(tp+fp))
        else:
            precision_fn.append(1.)
        recall_fn.append(tp/(tp+fn))
    precision_fp = []
    recall_fp = []
    for thr in np.unique(detected_loop):
        asd = detected_loop<=thr
        asd = asd & real_loop
        asd = asd & (distances <= 4)
        tp = asd.sum()
        fp = (detected_loop<=thr) & (distances > 4)
        fp = fp.sum()
        fn = (detected_loop > thr) & (real_loop)
        fn = fn.sum()
        if (tp+fp) > 0:
            precision_fp.append(tp/(tp+fp))
        else:
            precision_fp.append(1.)
        recall_fp.append(tp/(tp+fn))

    return precision_fn, recall_fn, precision_fp, recall_fp


def compute_PR_pairs(pair_dist, poses, is_distance=True, ignore_last=False, positive_range=4, ignore_below=-1):
    real_loop = []
    detected_loop = []
    distances = []
    last = poses.shape[0]
    if ignore_last:
        last = last-1
    for i in tqdm(range(100, last)):
        current_pose = poses[i][:3, 3]
        for j in range(i-50):
            candidate_pose = poses[j][:3, 3]
            dist_pose = np.linalg.norm(candidate_pose-current_pose)
            if dist_pose <= positive_range:
                real_loop.append(1)
            elif dist_pose <= ignore_below:
                continue
            else:
                real_loop.append(0)
            if is_distance:
                detected_loop.append(-pair_dist[i, j])
                distances.append(np.linalg.norm(current_pose - candidate_pose))
            else:
                detected_loop.append(pair_dist[i, j])
                distances.append(np.linalg.norm(current_pose - candidate_pose))


    precision, recall, _ = precision_recall_curve(real_loop, detected_loop)
    distances = np.array(distances)
    detected_loop = -np.array(detected_loop)
    real_loop = np.array(real_loop)
    precision2 = []
    recall2 = []
    for thr in np.unique(detected_loop):
        tp = detected_loop <= thr
        tp = tp & real_loop
        tp = tp & (distances <= 4)
        tp = tp.sum()
        fp = (detected_loop<thr).sum() - tp
        fn = (real_loop.sum()) - tp
        if (tp+fp) > 0.:
            precision2.append(tp/(tp+fp))
        else:
            precision2.append(1.)
        recall2.append(tp/(tp+fn))
    real_auc = auc(recall2, precision2)
    return precision, recall, precision2, recall2, real_auc


def compute_AP(precision, recall):
    ap = 0.
    for i in range(1, len(precision)):
        ap += (recall[i] - recall[i-1])*precision[i]
    return ap


def evaluate_model_with_emb(emb_list, datasets_list, positive_distance=5.):
    recall_sum = 0.
    start_idx = 0
    cont = 0
    F1_sum = 0.
    auc_sum = 0.
    auc_sum2 = 0.
    emb_list = emb_list.cpu().numpy()

    for dataset in datasets_list:
        poses = dataset.poses
        samples_num = len(dataset)
        finish_idx = start_idx + samples_num
        emb_sublist = emb_list[start_idx:finish_idx]

        recall, maxF1, wrong_auc, real_auc, recall2, precision2 = compute_recall(emb_sublist, poses, dataset, positive_distance)
        recall_sum = recall_sum + recall
        F1_sum += maxF1
        auc_sum += wrong_auc
        auc_sum2 += real_auc

        start_idx = finish_idx
        cont += 1

    final_recall = recall_sum / cont
    return final_recall, F1_sum / cont, auc_sum / cont, auc_sum2 / cont, recall2, precision2


def compute_recall(emb_list, poses, dataset, positive_distance=5.):
    print('compute_recall')
    have_matches = dataset.have_matches
    num_neighbors = 25
    recall_at_k = [0] * num_neighbors

    num_evaluated = 0
    emb_list = np.asarray(emb_list)

    for i in range(len(emb_list)):
        if hasattr(dataset, 'frames_with_gt'):
            if dataset.frames_with_gt[i] not in have_matches:
                continue
        elif i not in have_matches:
            continue
        min_range = max(0, i-50)
        max_range = min(i+50, len(emb_list))
        ignored_idxs = set(range(min_range, max_range))
        valid_idx = set(range(len(emb_list))) - ignored_idxs
        valid_idx = list(valid_idx)

        # tr = KDTree(emb_list[valid_idx])

        index = faiss.IndexFlatL2(emb_list.shape[1])
        index.add(emb_list[valid_idx])

        x = poses[i][0, 3]
        y = poses[i][1, 3]
        z = poses[i][2, 3]
        anchor_pose = torch.tensor([x, y, z])
        num_evaluated += 1
        # distances, indices = tr.query(np.array([emb_list[i]]), k=num_neighbors)

        D, I = index.search(emb_list[i:i+1], num_neighbors)

        indices = I[0]
        for j in range(len(indices)):

            m = valid_idx[indices[j]]
            x = poses[m][0, 3]
            y = poses[m][1, 3]
            z = poses[m][2, 3]
            possible_match_pose = torch.tensor([x, y, z])
            distance = torch.norm(anchor_pose - possible_match_pose)
            if distance <= positive_distance:
                # if j == 0:
                    # similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    # top1_similarity_score.append(similarity)
                recall_at_k[j] += 1
                break
    recall_at_k = (np.cumsum(recall_at_k) / float(num_evaluated)) * 100

    map_tree_poses = KDTree(np.stack(poses)[:, :3, 3])

    index = faiss.IndexFlatL2(emb_list.shape[1])
    index.add(emb_list[:50])

    real_loop = []
    detected_loop = []
    distances = []
    total_frame = 0
    for i in range(100, emb_list.shape[0]):
        min_range = max(0, i-50)  # Scan Context
        current_pose = torch.tensor(poses[i][:3, 3])

        indices = map_tree_poses.query_radius(np.expand_dims(current_pose, 0), positive_distance)
        valid_idxs = list(set(indices[0]) - set(range(min_range, emb_list.shape[0])))
        if len(valid_idxs) > 0:
            real_loop.append(1)
        else:
            real_loop.append(0)

        index.add(emb_list[i-50:i-49])
        nearest = index.search(emb_list[i:i+1], 1)

        total_frame += 1
        detected_loop.append(-nearest[0][0][0])
        candidate_pose = torch.tensor(poses[nearest[1][0][0]][:3, 3])
        distances.append((current_pose - candidate_pose).norm())


    precision, recall, _ = precision_recall_curve(real_loop, detected_loop)
    wrong_auc = average_precision_score(real_loop, detected_loop)
    F1 = [2*((precision[i]*recall[i])/(precision[i]+recall[i])) for i in range(len(precision))]

    distances = np.array(distances)
    detected_loop = -np.array(detected_loop)
    real_loop = np.array(real_loop)
    precision2 = []
    recall2 = []
    for thr in np.unique(detected_loop):
        tp = detected_loop <= thr
        tp = tp & real_loop
        tp = tp & (distances <= 4)
        tp = tp.sum()
        fp = (detected_loop<thr).sum() - tp
        fn = (real_loop.sum()) - tp
        if (tp+fp) > 0.:
            precision2.append(tp/(tp+fp))
        else:
            precision2.append(1.)
        recall2.append(tp/(tp+fn))
    real_auc = auc(recall2, precision2)

    return recall_at_k, np.array(F1).max(), wrong_auc, real_auc, recall2, precision2
