'''
    Based on D2-Net: A Trainable CNN for Joint Detection and Description of Local Features
    https://github.com/mihaidusmanu/d2-net
'''

import torch
from utils import fill_dustbins, min_row_col, interpolate_depth

MATCH_ID = 1
UNMATCH_ID = 0
IGNORED_ID = -1


def uv_to_pos(uv):
    return torch.cat([uv[0, :].view(1, -1), uv[1, :].view(1, -1)], dim=0)


def warp(pos1,
        depth1, intrinsics1, pose1, bbox1,
        depth2, intrinsics2, pose2, bbox2):
    device = pos1.device

    pos1 = torch.transpose(pos1, 0, 1)

    Z1, pos1, ids = interpolate_depth(pos1, depth1)
    # COLMAP convention
    u1 = pos1[0, :] + bbox1[1] + .5
    v1 = pos1[1, :] + bbox1[0] + .5

    X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
    Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])
    
    XYZ1_hom = torch.cat([
        X1.view(1, -1),
        Y1.view(1, -1),
        Z1.view(1, -1),
        torch.ones(1, Z1.size(0), device=device)
    ], dim=0)
    XYZ2_hom = torch.linalg.multi_dot([pose2, torch.inverse(pose1), XYZ1_hom])
    XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)
    
    uv2_hom = torch.matmul(intrinsics2, XYZ2)
    uv2 = uv2_hom[: -1, :] / uv2_hom[-1, :].view(1, -1)
    
    u2 = uv2[0, :] - bbox2[1] - .5
    v2 = uv2[1, :] - bbox2[0] - .5
    uv2 = torch.cat([u2.view(1, -1),  v2.view(1, -1)], dim=0)
    
    annotated_depth, pos2, new_ids = interpolate_depth(uv_to_pos(uv2), depth2)
    ids = ids[new_ids]

    pos1 = pos1[:, new_ids]
    estimated_depth = XYZ2[2, new_ids]
    
    inlier_mask = torch.abs(estimated_depth - annotated_depth)/annotated_depth < 0.1
   
    ids = ids[inlier_mask]
    pos2 = pos2[:, inlier_mask]
    pos1 = pos1[:, inlier_mask]
    return pos1, pos2, ids


def match_vectors(reprojected, original, match_threshold=3, unmatch_threshold=5):
    reproj_errors = None
    pairs, ignored = [], []

    if(reprojected.size(0) > 0):
        for i in range(original.size(0)):
            k1 = torch.unsqueeze(original[i], dim=0)
            dists = torch.cdist(k1, reprojected, p=2)
            
            if(reproj_errors == None):
                reproj_errors = dists
            else:
                reproj_errors = torch.cat([reproj_errors, dists], dim=0)

    if(reproj_errors != None):        
        for i in range(reproj_errors.size(0)):
            min_row, min_col = min_row_col(reproj_errors)
            if(reproj_errors[min_row, min_col].item() < match_threshold):
                pairs.append((min_row, min_col))
            elif(reproj_errors[min_row, min_col].item() < unmatch_threshold):
                ignored.append((min_row, min_col))
            else:
                break
            reproj_errors[min_row, :] = float("inf")
            reproj_errors[:, min_col] = float("inf")
    
    return pairs, ignored


def get_matches(kpts1, depth1, intrinsics1, pose1, bbox1, 
                kpts2, depth2, intrinsics2, pose2, bbox2):
                    
    matches = torch.full((kpts1.size(0)+1, kpts2.size(0)+1), UNMATCH_ID)
    kpts1_valid, kpts1_2, ids1 = warp(kpts1, depth1, intrinsics1, pose1, bbox1, depth2, intrinsics2, pose2, bbox2)
    kpts2_valid, kpts2_1, ids2 = warp(kpts2, depth2, intrinsics2, pose2, bbox2, depth1, intrinsics1, pose1, bbox1)
    
    kpts1_valid = torch.transpose(kpts1_valid, 0, 1)
    kpts2_1 = torch.transpose(kpts2_1, 0, 1)
    pairs1, ignored1 = match_vectors(kpts2_1, kpts1_valid, 3, 5)  
    
    kpts2_valid = torch.transpose(kpts2_valid, 0, 1)
    kpts1_2 = torch.transpose(kpts1_2, 0, 1)
    pairs2, ignored2 = match_vectors(kpts1_2, kpts2_valid, 3, 5)
    
    ids1 = ids1.cpu().numpy()
    ids2 = ids2.cpu().numpy()

    for pair in pairs1:
        if((pair[1], pair[0]) in pairs2):
            matches[ids1[pair[0]], ids2[pair[1]]] = MATCH_ID
            #print('matching ', ids1[pair[0]], ' -> ', ids2[pair[1]])

    for ignored in ignored1:
        matches[ids1[ignored[0]], ids2[ignored[1]]] = IGNORED_ID
        #print('ignoring ', ids1[ignored[0]], ' -> ', ids2[ignored[1]])

    for ignored in ignored2:
        matches[ids1[ignored[1]], ids2[ignored[0]]] = IGNORED_ID
        #print('ignoring ', ids1[ignored[1]], ' -> ', ids2[ignored[0]])
        
    matches = fill_dustbins(matches)

    return matches


def get_ground_truth(kpts1, kpts2, batch, device, plot_vis=False):
    batch_size = kpts1.size(0)
    batch_matches = None
    for idx_in_batch in range(batch_size):
        k1 = kpts1[idx_in_batch].to(device)
        depth1 = batch['depth1'][idx_in_batch].to(device)  # [h1, w1]
        intrinsics1 = batch['intrinsics1'][idx_in_batch].to(device)  # [3, 3]
        pose1 = batch['pose1'][idx_in_batch].view(4, 4).to(device)  # [4, 4]
        bbox1 = batch['bbox1'][idx_in_batch].to(device)  # [2]
        
        k2 = kpts2[idx_in_batch].to(device)
        depth2 = batch['depth2'][idx_in_batch].to(device)
        intrinsics2 = batch['intrinsics2'][idx_in_batch].to(device)
        pose2 = batch['pose2'][idx_in_batch].view(4, 4).to(device)
        bbox2 = batch['bbox2'][idx_in_batch].to(device)

        matches = get_matches(k1, depth1, intrinsics1, pose1, bbox1, 
                              k2, depth2, intrinsics2, pose2, bbox2)
        matches = torch.unsqueeze(matches, dim = 0)
        
        if(batch_matches != None):
            batch_matches = torch.cat((batch_matches, matches), dim=0)
        else:
            batch_matches = matches

    return batch_matches