import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import open3d.core as o3c
import numpy as np


class PointNetHead(nn.Module):

    def __init__(self, input_dim, points_num, rotation_parameters=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.FC1 = nn.Linear(input_dim * 2, 1024)
        self.FC2 = nn.Linear(1024, 512)
        self.FC_transl = nn.Linear(512, 3)
        self.FC_rot = nn.Linear(512, rotation_parameters)

    def forward(self, x, compute_transl=True, compute_rotation=True):
        x = x.view(x.shape[0], -1)

        x = self.relu(self.FC1(x))
        x = self.relu(self.FC2(x))
        if compute_transl:
            transl = self.FC_transl(x)
        else:
            transl = None

        if compute_rotation:
            yaw = self.FC_rot(x)
        else:
            yaw = None
        # print(transl, yaw)
        return transl, yaw

def compute_rigid_transform_mask(points1_stack, points2_stack, weights_stack, mask_ac, mask_A, mask_po):
    """Compute rigid transforms between two point clouds via weighted SVD.
       Adapted from https://github.com/yewzijian/RPMNet/
    Args:
        points1 (torch.Tensor): (B, M, 3) coordinates of the first point cloud
        points2 (torch.Tensor): (B, N, 3) coordinates of the second point cloud
        weights (torch.Tensor): (B, M)
    Returns:
        Transform T (B, 3, 4) to get from points1 to points2, i.e. T*points1 = points2
    """
    transform_stack = torch.zeros(points1_stack.shape[0],3,4).cuda()

    for i in range(points1_stack.shape[0]):
        points1 = points1_stack[i:i+1,:,:]
        points1 = points1[mask_ac[i:i+1,:,:]].view(1,-1,3)
        points2 = points2_stack[i:i+1,:,:]
        points2 = points2[mask_ac[i:i+1,:,:]].view(1,-1,3)
        weights = weights_stack[i:i+1,:,:]
        weights = weights[mask_A[i:i+1].unsqueeze(-1)].view(1,-1)

        weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-5)
        centroid_a = torch.sum(points1 * weights_normalized, dim=1)
        centroid_b = torch.sum(points2 * weights_normalized, dim=1)
        a_centered = points1 - centroid_a[:, None, :]
        b_centered = points2 - centroid_b[:, None, :]
        cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

        # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
        # and choose based on determinant to avoid flips
        u, s, v = torch.svd(cov, some=False, compute_uv=True)
        rot_mat_pos = v @ u.transpose(-1, -2)
        v_neg = v.clone()
        v_neg[:, :, 2] *= -1
        rot_mat_neg = v_neg @ u.transpose(-1, -2)
        rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
        assert torch.all(torch.det(rot_mat) > 0)

        # Compute translation (uncenter centroid)
        translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

        transform = torch.cat((rot_mat, translation), dim=2)
        transform_stack[i:i+1,:,:] = transform
    return transform_stack

def compute_rigid_transform(points1, points2, weights):
    """Compute rigid transforms between two point clouds via weighted SVD.
       Adapted from https://github.com/yewzijian/RPMNet/
    Args:
        points1 (torch.Tensor): (B, M, 3) coordinates of the first point cloud
        points2 (torch.Tensor): (B, N, 3) coordinates of the second point cloud
        weights (torch.Tensor): (B, M)
    Returns:
        Transform T (B, 3, 4) to get from points1 to points2, i.e. T*points1 = points2
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-5)
    centroid_a = torch.sum(points1 * weights_normalized, dim=1)
    centroid_b = torch.sum(points2 * weights_normalized, dim=1)
    a_centered = points1 - centroid_a[:, None, :]
    b_centered = points2 - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # rot_mat = 

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    return transform


def sinkhorn_slack_variables(feature1, feature2, beta, alpha, n_iters = 5, slack = True):
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix using slack variables (dustbins)
        Adapted from https://github.com/yewzijian/RPMNet/
    Args:
        feature1 (torch.Tensor): point-wise features of the first point cloud.
        feature2 (torch.Tensor): point-wise features of the second point cloud.
        beta (torch.Tensor): annealing parameter.
        alpha (torch.Tensor): matching rejection parameter.
        n_iters (int): Number of normalization iterations.
        slack (bool): Whether to include slack row and column.
    Returns:
        log(perm_matrix): (B, J, K) Doubly stochastic matrix.
    """

    B, N, _ = feature1.shape
    _, M, _ = feature2.shape

    # Feature normalization
    feature1 = feature1 / feature1.norm(dim=-1, keepdim=True)
    feature2 = feature2 / feature2.norm(dim=-1, keepdim=True)

    dist = -2 * torch.matmul(feature1, feature2.permute(0, 2, 1))
    dist += torch.sum(feature1 ** 2, dim=-1)[:, :, None]
    dist += torch.sum(feature2 ** 2, dim=-1)[:, None, :]

    log_alpha = -beta * (dist - alpha)

    # Sinkhorn iterations
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

    return log_alpha


def sinkhorn_unbalanced(feature1, feature2, epsilon, gamma, max_iter):
    """
    Sinkhorn algorithm for Unbalanced Optimal Transport.
    Modified from https://github.com/valeoai/FLOT/
    Args:
        feature1 (torch.Tensor):
            (B, N, C) Point-wise features for points cloud 1.
        feature2 (torch.Tensor):
            (B, M, C) Point-wise features for points cloud 2.
        epsilon (torch.Tensor):
            Entropic regularization.
        gamma (torch.Tensor):
            Mass regularization.
        max_iter (int):
            Number of iteration of the Sinkhorn algorithm.
    Returns:
        T (torch.Tensor):
            (B, N, M) Transport plan between point cloud 1 and 2.
    """

    # Transport cost matrix
    feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))

    # Entropic regularisation
    K = torch.exp(-C / epsilon) #* support

    # Early return if no iteration
    if max_iter == 0:
        return K

    # Init. of Sinkhorn algorithm
    power = gamma / (gamma + epsilon)
    a = (
            torch.ones(
                (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
            )
            / K.shape[1]
    )
    prob1 = (
            torch.ones(
                (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
            )
            / K.shape[1]
    )
    prob2 = (
            torch.ones(
                (K.shape[0], K.shape[2], 1), device=feature2.device, dtype=feature2.dtype
            )
            / K.shape[2]
    )

    # Sinkhorn algorithm
    for _ in range(max_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))

    return T

def sinkhorn_unbalanced_overlap(feature1, feature2, epsilon, gamma, max_iter, mask_A, mask_B):
    """
    Sinkhorn algorithm for Unbalanced Optimal Transport.
    Modified from https://github.com/valeoai/FLOT/
    Args:
        feature1 (torch.Tensor):
            (B, N, C) Point-wise features for points cloud 1.
        feature2 (torch.Tensor):
            (B, M, C) Point-wise features for points cloud 2.
        epsilon (torch.Tensor):
            Entropic regularization.
        gamma (torch.Tensor):
            Mass regularization.
        max_iter (int):
            Number of iteration of the Sinkhorn algorithm.
    Returns:
        T (torch.Tensor):
            (B, N, M) Transport plan between point cloud 1 and 2.
    """

    # Transport cost matrix
    feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))

    # Entropic regularisation
    K = torch.exp(-C / epsilon) #* support

    # Early return if no iteration
    if max_iter == 0:
        return K

    # Init. of Sinkhorn algorithm
    power = gamma / (gamma + epsilon)
    a = (
            torch.ones(
                (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
            )
            / K.shape[1]
    )
    prob1 = (torch.ones((K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype)/ K.shape[1])
    prob2 = (torch.ones((K.shape[0], K.shape[2], 1), device=feature2.device, dtype=feature2.dtype)/ K.shape[2])

    for _ in range(max_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))
    mask_A = mask_A.unsqueeze(0).expand(K.shape[2],K.shape[0],K.shape[1]).permute(1,2,0)
    mask_B = mask_B.unsqueeze(0).expand(K.shape[1],K.shape[0],K.shape[2]).permute(1,0,2)


    mask_T = torch.zeros(T.shape[0],T.shape[1],T.shape[2], device=T.device)
    mask_T[torch.logical_and((mask_A),(mask_B))] = 1
    # mask_T[torch.logical_or((mask_A),(mask_B))] = 1
    
    # mask_T_pone = torch.zeros(T.shape[0],T.shape[1],T.shape[2], device=T.device)
    # mask_T_pone[~mask_B] = 1
    
    # mask_T_nepo = torch.zeros(T.shape[0],T.shape[1],T.shape[2], device=T.device)
    # mask_T_nepo[~mask_A] = 1
    
    # mask_T_nene = torch.zeros(T.shape[0],T.shape[1],T.shape[2], device=T.device)
    # mask_T_nene[torch.logical_and((~mask_A),(~mask_B))] = 1

    # return T*mask_T , torch.sum(torch.mul(T*mask_T_pone,C), (1,2)), torch.sum(torch.mul(T*mask_T_nepo,C), (1,2)) \
    #         , torch.sum(torch.mul(T*mask_T_nene,C), (1,2))
    return T, mask_T 

def sinkhorn_unbalanced_overlap2(feature1, feature2, epsilon, gamma, max_iter, mask_A, mask_B):

    # Transport cost matrix
    feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))

    # Entropic regularisation
    K = torch.exp(-C / epsilon) #* support

    # Early return if no iteration
    if max_iter == 0:
        return K

    # Init. of Sinkhorn algorithm
    power = gamma / (gamma + epsilon)
    a = (
            torch.ones(
                (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
            )
            / K.shape[1]
    )


    prob1 = (( torch.ones((K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype) - 0.00004096)   / 
             torch.count_nonzero(mask_A, dim=1).unsqueeze(1).expand(-1,K.shape[1]).unsqueeze(2)  )
    prob1 = prob1*mask_A.unsqueeze(2) + 1e-8
    prob2 = (( torch.ones((K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype) - 0.00004096)   / 
             torch.count_nonzero(mask_A, dim=1).unsqueeze(1).expand(-1,K.shape[1]).unsqueeze(2))
    prob2 = prob2*mask_B.unsqueeze(2) + 1e-8

    # print(prob1)
    # print(prob2)

    for _ in range(max_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        # print(KTa)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # print(b)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)
        # print(a)
        

    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))
    # print(T)
    return T 


# def draw_registration_result(source, target, source_ne, target_ne, transformation):
#     source_temp = source.clone()
#     target_temp = target.clone()
#     source_ne_temp = source_ne.clone()
#     target_ne_temp = target_ne.clone()

#     source_temp.transform(transformation)
#     source_ne_temp.transform(transformation)

#     # This is patched version for tutorial rendering.
#     # Use `draw` function for you application.
#     o3d.visualization.draw_geometries(
#         [source_temp.to_legacy(),
#          target_temp.to_legacy()],
#         zoom=0.4459,
#         front=[0.9288, -0.2951, -0.2242],
#         lookat=[1.6784, 2.0612, 1.4451],
#         up=[-0.3402, -0.9189, -0.1996])
    
#     o3d.visualization.draw_geometries(
#         [source_ne_temp.to_legacy(),
#          target_ne_temp.to_legacy()],
#         zoom=0.4459,
#         front=[0.9288, -0.2951, -0.2242],
#         lookat=[1.6784, 2.0612, 1.4451],
#         up=[-0.3402, -0.9189, -0.1996])
    
#     o3d.visualization.draw_geometries(
#         [target_ne_temp.to_legacy(),
#          target_temp.to_legacy()],
#         zoom=0.4459,
#         front=[0.9288, -0.2951, -0.2242],
#         lookat=[1.6784, 2.0612, 1.4451],
#         up=[-0.3402, -0.9189, -0.1996])
    


def overap_mask(A, B, delta_pose):
    batch, N_A, C = A.shape
    _, N_B, _ = B.shape

    A_color = torch.tensor([1,0,0], dtype=torch.float32)
    A_color = A_color.expand(batch,N_A,3)
    B_color = torch.tensor([0,1,0], dtype=torch.float32)
    B_color = B_color.expand(batch,N_B,3)

    mask_A = torch.zeros([batch, N_A], dtype=torch.bool).cuda()
    mask_B = torch.zeros([batch, N_B], dtype=torch.bool).cuda()
    mask_distance = torch.zeros([batch, N_A, N_B], dtype=torch.float32).cuda()
    rotation = []

    for i in range(batch):
        batch_A = A[i:i+1].squeeze()
        batch_B = B[i:i+1].squeeze()

        source = o3d.t.geometry.PointCloud().cuda()
        source.point.positions = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(batch_A))
        target = o3d.t.geometry.PointCloud().cuda()
        target.point.positions = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(batch_B))
        source.point.colors = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(A_color[i:i+1].squeeze()))
        target.point.colors = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(B_color[i:i+1].squeeze()))
        
        max_correspondence_distance = 0.5
        init_source_to_target = delta_pose[i:i+1].squeeze().detach().cpu().numpy()
        
        estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()

        criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)

        registration_icp = o3d.t.pipelines.registration.icp(source, target, max_correspondence_distance,
                                    init_source_to_target, estimation, criteria)
        temp = torch.utils.dlpack.from_dlpack(registration_icp.correspondences_.to_dlpack()).transpose(0,1).cuda()
        
        mask_A[i:i+1,:] = (temp != -1)
        temp[temp==-1] = 0
        mask_B[i:i+1,:] = torch.zeros(1, N_B).cuda().scatter_(1, temp, True)
        rotation.append(registration_icp.transformation)

        transform = torch.utils.dlpack.from_dlpack(registration_icp.transformation.to_dlpack()).cuda()
        transform = transform.float()

        batch_A = torch.cat((batch_A, torch.zeros(N_A, 1).cuda()), dim=1)
        batch_A[:, 3] = 1.
        batch_A = (transform @ batch_A.T).T
        batch_A = batch_A[:,:3]

        euclidean_distance = torch.cdist(batch_A,batch_B)
        mask_distance[i:i+1,:,:] = torch.exp(-euclidean_distance)*0.5 + 1.5
    
    # print(mask_distance)
    # number_A = torch.count_nonzero(mask_A, dim=1)
    # number_B = torch.count_nonzero(mask_B, dim=1)
    # A_max, A_MAX_index = number_A.max(dim=0)
    # B_max, B_MAX_index = number_B.max(dim=0)
    # for j in range(batch):
    #     if j != A_MAX_index and number_A[j] != A_max:
    #         A_nonzero = torch.nonzero(~mask_A[j]).squeeze()
    #         A_index = A_nonzero[torch.ones([A_nonzero.shape[0]]).multinomial(A_max.item() - number_A[j].item())]
    #         mask_A[j:j+1,A_index] = True
    #     if j != B_MAX_index and number_B[j] != B_max:
    #         B_nonzero = torch.nonzero(~mask_B[j]).squeeze()
    #         B_index = B_nonzero[torch.ones([B_nonzero.shape[0]]).multinomial(B_max - number_B[j])]
    #         mask_B[j:j+1,B_index] = True
    

    mask_ac = mask_A.expand(3,batch,N_A).permute(1, 2, 0)
    mask_po = mask_B.expand(3,batch,N_B).permute(1, 2, 0)
    
    # print(mask_ac.shape)
    # number_A = torch.count_nonzero(mask_A, dim=1)
    # number_B = torch.count_nonzero(mask_B, dim=1) 
    # print(number_A)
    # print(number_B)
    # for i in range(batch):
    # ## visualize
    #     atemp = A[i:i+1]
    #     btemp = B[i:i+1]
    #     mask_pointapo = atemp[mask_po[i:i+1]].view(1,-1,3)
    #     mask_pointane = atemp[~mask_po[i:i+1]].view(1,-1,3)
    #     mask_pointbpo = btemp[mask_ac[i:i+1]].view(1,-1,3)
    #     mask_pointbne = btemp[~mask_ac[i:i+1]].view(1,-1,3)
    #     print(mask_pointapo.shape)
    #     print(mask_pointane.shape)
    #     print(mask_pointbpo.shape)
    #     print(mask_pointbne.shape)
    #     _, N_A, C = mask_pointapo.shape
    #     A_color = torch.tensor([1,0,0], dtype=torch.float32).cuda()
    #     A_color = A_color.expand(1,N_A,3)
        
    #     _, N_B, C = mask_pointbpo.shape
    #     B_color = torch.tensor([0,1,0], dtype=torch.float32).cuda()
    #     B_color = B_color.expand(1,N_B,3)
    #     source = o3d.t.geometry.PointCloud().cuda()
    #     source.point.positions = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(mask_pointapo.squeeze()))
    #     source.point.colors = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(A_color.squeeze()))
    #     target = o3d.t.geometry.PointCloud().cuda()
    #     target.point.positions = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(mask_pointbpo.squeeze()))
    #     target.point.colors = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(B_color.squeeze()))

    #     _, N_A, C = mask_pointane.shape
    #     A_color = torch.tensor([0,0,1], dtype=torch.float32).cuda()
    #     A_color = A_color.expand(1,N_A,3)
        
    #     _, N_B, C = mask_pointbne.shape
    #     B_color = torch.tensor([0,0,0], dtype=torch.float32).cuda()
    #     B_color = B_color.expand(1,N_B,3)
    #     source_ne = o3d.t.geometry.PointCloud().cuda()
    #     source_ne.point.positions = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(mask_pointane.squeeze()))
    #     source_ne.point.colors = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(A_color.squeeze()))
    #     target_ne = o3d.t.geometry.PointCloud().cuda()
    #     target_ne.point.positions = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(mask_pointbne.squeeze()))
    #     target_ne.point.colors = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(B_color.squeeze()))
    
    #     draw_registration_result(source, target, source_ne, target_ne, rotation[i])
        
    
    ## mask_A = anchor's overlap index [batch, N(4096)] , boolean
    ## mask_B = positive's overlap index [batch, N(4096)] , boolean
    ## mask_ac = anchor's overlap 3d coor [batch, N(4096), 3 ] , boolean
    ## mask_po = anchor's overlap 3d coor [batch, N(4096), 3 ] , boolean
    return mask_A, mask_B, mask_ac, mask_po, mask_distance

class UOTHead(nn.Module):

    def __init__(self, nb_iter=5, use_svd=False, sinkhorn_type='unbalanced'):
        super().__init__()
        self.epsilon = torch.nn.Parameter(torch.zeros(1))  # Entropic regularisation
        self.gamma = torch.nn.Parameter(torch.zeros(1))  # Mass regularisation
        self.nb_iter = nb_iter
        self.use_svd = use_svd
        self.sinkhorn_type = sinkhorn_type

        if not use_svd:
            raise NotImplementedError()

    def forward(self, batch_dict, delta_pose, compute_transl=True, compute_rotation=True, src_coords=None, mode='pairs', is_training=True):

        feats = batch_dict['point_features'].squeeze(-1)
        B, C, NUM = feats.shape

        assert B % 2 == 0, "Batch size must be multiple of 2: B anchor + B positive samples"
        B = B // 2
        feat1 = feats[:B]
        feat2 = feats[B:]

        coords = batch_dict['point_coords'].view(2*B, -1, 4)
        coords1 = coords[:B, :, 1:]
        coords2 = coords[B:, :, 1:]
        mask_A, mask_B, mask_ac, mask_po, mask_distance = overap_mask(coords1, coords2, delta_pose)
       
        if is_training == False:
            
            mask_A[:,:] = 1
            mask_B[:,:] = 1 
            mask_ac[:,:,:] = 1
            mask_po[:,:,:] = 1
            mask_distance[:,:,:] = 1
            # print(mask_distance)
            # print(mask_ac)

        if self.sinkhorn_type == 'unbalanced':
            transport, mask_T = sinkhorn_unbalanced_overlap(
                feat1.permute(0, 2, 1),
                feat2.permute(0, 2, 1),
                epsilon=torch.exp(self.epsilon) + 0.03,
                gamma=torch.exp(self.gamma),
                max_iter=self.nb_iter,
                mask_A = mask_A,
                mask_B = mask_B
            )
            transport_unmask = transport
            # transport_mask = transport*mask_T
            transport_mask = transport*mask_distance*mask_T
            # transport = sinkhorn_unbalanced_overlap2(
            #     feat1.permute(0, 2, 1),
            #     feat2.permute(0, 2, 1),
            #     epsilon=torch.exp(self.epsilon) + 0.03,
            #     gamma=torch.exp(self.gamma),
            #     max_iter=self.nb_iter,
            #     mask_A = mask_A,
            #     mask_B = mask_B
            # )
            # print(transport)
            # transport_unmask = transport
            # transport_mask = transport
            # print(mask_distance)
        else:
            transport = sinkhorn_slack_variables(
                feat1.permute(0, 2, 1),
                feat2.permute(0, 2, 1),
                F.softplus(self.epsilon),
                F.softplus(self.gamma),
                self.nb_iter,
            )
            transport = torch.exp(transport)

        row_sum_mask = transport_mask.sum(-1, keepdim=True)
        row_sum_unmask = transport_unmask.sum(-1, keepdim=True)

        # Compute the "projected" coordinates of point cloud 1 in
        # point cloud 2 based on the optimal transport plan
        sinkhorn_matches_mask = (transport_mask @ coords2) / (row_sum_mask + 1e-8)
        sinkhorn_matches_unmask  = (transport_unmask @ coords2) / (row_sum_unmask + 1e-8)
        
        batch_dict['sinkhorn_matches_mask'] = sinkhorn_matches_mask
        batch_dict['sinkhorn_matches_unmask'] = sinkhorn_matches_unmask
        batch_dict['transport'] = transport_mask
        batch_dict['mask_ac'] = mask_ac
        batch_dict['mask_po'] = mask_po
        # batch_dict['ac2po_ne_distance'] = ac2po_ne_distance
        # batch_dict['ac_ne2po_distance'] = ac_ne2po_distance
        # batch_dict['ac_ne2po_ne_distance'] = ac_ne2po_ne_distance

        if not self.use_svd:
            raise NotImplementedError()
        else:
            if src_coords is None:
                src_coords = coords1
            # transformation = compute_rigid_transform(src_coords[mask_ac].view(B,-1,3), sinkhorn_matches[mask_ac].view(B,-1,3), row_sum[mask_A.unsqueeze(-1)].view(B,-1))
            # transformation_mask = compute_rigid_transform_mask(src_coords, sinkhorn_matches_mask, row_sum_mask , mask_ac, mask_A, mask_po)
            transformation_unmask = compute_rigid_transform(src_coords, sinkhorn_matches_unmask, row_sum_unmask.squeeze(-1))
            batch_dict['transformation'] = transformation_unmask
            batch_dict['transformation_mask'] = transformation_unmask
            # batch_dict['point_ac_mask'] = mask_ac
            batch_dict['out_rotation'] = None
            batch_dict['out_translation'] = None

        return batch_dict