import cv2
import torch
import numpy as np
from collections import defaultdict
import torch.nn.functional as F


def preprocess_image(image, preprocessing=None):
    image = image.astype(np.float32)
    image = np.moveaxis(image, -1, 0)
    #image = np.expand_dims(image, 0)
    if preprocessing == 'caffe':
        # RGB -> BGR
        image = image[:: -1, :, :]
        # Zero-center by mean pixel
        mean = np.array([103.939, 116.779, 123.68])
        image = image - mean.reshape([3, 1, 1])
    else:
        pass
    return image


def create_kpts_image(img, kpts, color=(255,255,255)):
    for k in kpts:
        img = cv2.circle(img, (int(k[0]), int(k[1])), 3, color, 2)
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