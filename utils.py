import cv2
import torch
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.cm as cm
from models.utils import make_matching_plot_fast


def preprocess_image(image):
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)
    return image


def create_kpts_image(img, kpts, color=(255,255,255)):
    for k in kpts:
        img = cv2.circle(img, (int(k[0]), int(k[1])), 3, color, 2)
    return img


def create_matches_image(img0, img1, kpts0, kpts1, matches, scores):
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = scores[valid]
    color = cm.jet(mconf)
    text = ['SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),]
    
    img = make_matching_plot_fast(img0, img1, kpts0, kpts1,
                                  mkpts0, mkpts1, color, text, 
                                  show_keypoints=True)
    return img



def pad_data(data, max_kpts, img_shape, device):
    _, _, width, _ = img_shape

    for k in data:
        if isinstance(data[k], (list, tuple)):
            new_data = []
            if(k.startswith('keypoints')):
                #padding keypoints
                for kpt in data[k]:
                    #random_values = torch.Tensor(max_kpts - kpt.shape[0], 2).uniform_(0, width)
                    random_values = torch.randint(0, width, (max_kpts - kpt.shape[0], 2))
                    new_data += [torch.cat((kpt, random_values.to(device)), 0)]
                    
            if(k.startswith('descriptor')):
                #padding descriptors
                for desc in data[k]:
                    new_data += [F.pad(desc, 
                                (0, max_kpts - desc.shape[1]))]

            if(k.startswith('score')):
                #padding scores
                for score in data[k]:
                    new_data += [F.pad(score, 
                                (0, max_kpts - score.shape[0]))]
            data[k] = torch.stack(new_data)
    return data
    

def replace_ignored(data, ignore, img_shape, device):
    _, _, width, _ = img_shape

    for img_id in ['0', '1']:
        for k in data:
            batch_size = data[k].size(0)
            if(k.startswith('keypoints'+img_id)):
                for i in range(batch_size):
                    for id in ignore['ignored'+img_id][i]:
                        new_row = torch.randint(0, width, (1, 2))
                        data[k][i][id] = new_row
            if(k.startswith('score'+img_id)):
                for i in range(batch_size):        
                    for id in ignore['ignored'+img_id][i]:
                        data[k][i][id] = 0
    return data


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def min_row_col(tensor):
    i = 0
    smallest, min_i, min_j = None, None, None
    for row in tensor:
        min_value = torch.min(row)
        if(smallest == None or min_value < smallest):
            smallest = min_value
            min_i = i
            min_j = torch.argmin(row).item()
        i += 1

    return min_i, min_j


def get_only_balanced(data, gt, max_kpts):
    new_data = defaultdict(lambda: None)
    new_gt = None
    for i in range(gt.size(0)):
        valid_ids = (gt[i] != -1).nonzero(as_tuple=True)
        filtered_target = gt[i][valid_ids]
        pos_ids = (filtered_target < max_kpts).nonzero(as_tuple=True)
        neg_ids = (filtered_target == max_kpts).nonzero(as_tuple=True) 
        total_size = len(pos_ids[0])+len(neg_ids[0])
        
        if(len(pos_ids[0])/total_size > 0.5):
            if(new_gt == None):
                new_gt = torch.unsqueeze(gt[i], dim=0)
            else:
                new_gt = torch.cat((new_gt, torch.unsqueeze(gt[i], dim=0)), dim=0)
            
            for k in data:
                if(new_data[k] == None):
                    new_data[k] = torch.unsqueeze(data[k][i], dim=0)
                else:
                    new_data[k] = torch.cat((new_data[k], torch.unsqueeze(data[k][i], dim=0)), dim=0)
    return new_data, new_gt


def fill_dustbins(matches):
    rows = torch.count_nonzero(matches, dim=1)
    cols = torch.count_nonzero(matches, dim=0)
    dust_col = rows.clone()
    dust_row = cols.clone()
    dust_col[rows == 0] = 1
    dust_col[rows != 0] = 0
    dust_row[cols == 0] = 1
    dust_row[cols != 0] = 0
    matches[:,-1] = dust_col
    matches[-1,:] = dust_row
    return matches


def ohe_to_le(ohe_tensor):
    '''
        Function to convert one hot encoding to label encoding. Notice that if all elements in a row/cols are zero, the keypoint has no match, 
        thus its label is assigned to n_rows/n_cols. MOreover, if the keypoint is ignored its label is assigned to -1
    '''
    le_tensor = torch.full((ohe_tensor.size(0), ohe_tensor.size(-1)), ohe_tensor.size(-1))
    match_ids = (ohe_tensor == 1).nonzero(as_tuple=True)
    ignored_ids = (ohe_tensor == -1).nonzero(as_tuple=True)    
    le_tensor[match_ids[:2]] = match_ids[2]
    le_tensor[ignored_ids[:2]] = -1
    
    return le_tensor


def save_model(path, model, optimizer, step, epoch, loss):
    torch.save({'epoch': epoch,
                'step': step,
                'kenc': model.kenc.state_dict(),
                'gnn': model.gnn.state_dict(),
                'final_proj': model.final_proj.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss}, 
                path)
    print(f'Model {path} saved!')


def load_model(model, path):
    print('Loading model ', path)
    ckpt = torch.load(str(path))
    model.load_state_dict(ckpt)
    return model


def load_model_weights(model, path, recover_state=False, modules=['gnn', 'final_proj']):
    print('Loading model ', path)
    ckpt = torch.load(str(path))
    if('kenc' in modules):
        model.kenc.load_state_dict(ckpt['kenc'])
    if('gnn' in modules):
        model.gnn.load_state_dict(ckpt['gnn'])
    if('final_proj' in modules):
        model.final_proj.load_state_dict(ckpt['final_proj'])
    if(recover_state):
        return model, ckpt['epoch'], ckpt['step'], ckpt['optimizer'], ckpt['loss']
    return model


def scores_to_matches(scores, threshold=0.5):
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    valid0 = mutual0 & (mscores0 > threshold)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    
    return indices0, mscores0



def interpolate_depth(pos, depth):

    device = pos.device
    ids = torch.arange(0, pos.size(1), device=device)
    h, w = depth.size()
    
    i = pos[1, :]
    j = pos[0, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]

    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left
    
    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([j.view(1, -1), i.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]
