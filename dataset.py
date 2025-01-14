'''
    Based on D2-Net: A Trainable CNN for Joint Detection and Description of Local Features
    https://github.com/mihaidusmanu/d2-net
'''

import os
import torch
import cv2
import h5py
import pickle
import numpy as np
from skimage import io
from torch.utils.data import Dataset
from tqdm import tqdm
from skimage.transform import resize
from utils import preprocess_image


class MegaDepthDataset(Dataset):
    def __init__(self,
                 scene_list_path='megadepth_utils/train_scenes.txt',
                 scene_info_path='/local/dataset/megadepth/scene_info',
                 base_path='/local/dataset/megadepth',
                 train=True,
                 preprocessing=None,
                 min_overlap_ratio=.5,
                 max_overlap_ratio=1,
                 max_scale_ratio=np.inf,
                 pairs_per_scene=100,
                 image_size=256,
                 save_dataset_path='/local/dataset/megadepth/dataset.pkl',
                 load_from_file=False):
        self.scenes = []
        with open(scene_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.scenes.append(line.strip('\n'))
        
        self.scene_info_path = scene_info_path
        self.base_path = base_path

        self.train = train

        self.preprocessing = preprocessing

        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.max_scale_ratio = max_scale_ratio

        self.pairs_per_scene = pairs_per_scene

        self.image_size = image_size
        self.dataset = []
        self.save_path = save_dataset_path
        self.load_from_file = load_from_file
        if(load_from_file):
            self.load_path = save_dataset_path
    

    def file_is_valid(self, file_path):
        return (os.path.exists(file_path) and os.path.getsize(file_path) > 1000) #1kb
        
    
    def build_dataset(self):
        self.dataset = []
       
        if(self.load_from_file):
            if not self.train:
                print('Loading validation dataset')
            else:
                print('Loading training dataset')    
            with open(self.load_path, "rb") as f:
                self.dataset = pickle.load(f)
        else:
            if not self.train:
                np_random_state = np.random.get_state()
                np.random.seed(42)
                print('Building the validation dataset...')
            else:
                print('Building a new training dataset...')

            for scene in tqdm(self.scenes, total=len(self.scenes)):
                scene_info_path = os.path.join(
                    self.scene_info_path, '%s.0.npz' % scene
                )
                if not os.path.exists(scene_info_path):
                    continue
                scene_info = np.load(scene_info_path, allow_pickle=True)
                overlap_matrix = scene_info['overlap_matrix']
                scale_ratio_matrix = scene_info['scale_ratio_matrix']
                valid =  np.logical_and(np.logical_and(overlap_matrix >= self.min_overlap_ratio,
                                                    overlap_matrix <= self.max_overlap_ratio),
                                                        scale_ratio_matrix <= self.max_scale_ratio)
                
                pairs = np.vstack(np.where(valid))
                try:
                    selected_ids = np.random.choice(pairs.shape[1], self.pairs_per_scene)
                except:
                    continue
                
                image_paths = scene_info['image_paths']
                depth_paths = scene_info['depth_paths']
                points3D_id_to_2D = scene_info['points3D_id_to_2D']
                points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']
                intrinsics = scene_info['intrinsics']
                poses = scene_info['poses']
                
                for pair_idx in selected_ids:
                    idx1 = pairs[0, pair_idx]
                    idx2 = pairs[1, pair_idx]
                    matches = np.array(list(points3D_id_to_2D[idx1].keys() &
                                            points3D_id_to_2D[idx2].keys()))

                    # Scale filtering
                    matches_nd1 = np.array([points3D_id_to_ndepth[idx1][match] for match in matches])
                    matches_nd2 = np.array([points3D_id_to_ndepth[idx2][match] for match in matches])
                    scale_ratio = np.maximum(matches_nd1 / matches_nd2, matches_nd2 / matches_nd1)
                    matches = matches[np.where(scale_ratio <= self.max_scale_ratio)[0]]
                    
                    point3D_id = np.random.choice(matches)
                    point2D1 = points3D_id_to_2D[idx1][point3D_id]
                    point2D2 = points3D_id_to_2D[idx2][point3D_id]
                    nd1 = points3D_id_to_ndepth[idx1][point3D_id]
                    nd2 = points3D_id_to_ndepth[idx2][point3D_id]
                    central_match = np.array([
                        point2D1[1], point2D1[0],
                        point2D2[1], point2D2[0]
                    ])

                    image_path1 = os.path.join(self.base_path, image_paths[idx1])
                    image_path2 = os.path.join(self.base_path, image_paths[idx2])
                    depth_path1 = os.path.join(self.base_path, 'data',
                                            depth_paths[idx1].replace('phoenix/S6/zl548/MegaDepth_v1/', ''))
                    depth_path2 = os.path.join(self.base_path, 'data',
                                            depth_paths[idx2].replace('phoenix/S6/zl548/MegaDepth_v1/', ''))
                    
                    
                    if(self.file_is_valid(image_path1) and self.file_is_valid(image_path2) and  
                       self.file_is_valid(depth_path1) and self.file_is_valid(depth_path2)):
                        self.dataset.append({
                            'image_path1': image_path1,
                            'depth_path1': depth_path1,
                            'intrinsics1': intrinsics[idx1],
                            'pose1': poses[idx1],
                            'image_path2': image_path2,
                            'depth_path2': depth_path2,
                            'intrinsics2': intrinsics[idx2],
                            'pose2': poses[idx2],
                            'central_match': central_match,
                            'scale_ratio': max(nd1 / nd2, nd2 / nd1)
                        })

            np.random.shuffle(self.dataset)
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.dataset, f)

            if not self.train:
                np.random.set_state(np_random_state)
    
    def __len__(self):
        return len(self.dataset)
   
    def recover_pair(self, pair_metadata):
        depth_path1 = pair_metadata['depth_path1']
        
        with h5py.File(depth_path1, 'r') as hdf5_file:
            depth1 = np.array(hdf5_file['/depth'])
        assert(np.min(depth1) >= 0)
        
        image_path1 = pair_metadata['image_path1']

        image1 = io.imread(image_path1, as_gray=True)
        assert(image1.shape[0] == depth1.shape[0] and image1.shape[1] == depth1.shape[1])
        intrinsics1 = pair_metadata['intrinsics1']
        pose1 = pair_metadata['pose1']

        #resize so shortest dimension is fixed in image_size
        original1_shape = image1.shape
        new_size1 = np.asarray(image1.shape[:2]/np.min(image1.shape[:2])*self.image_size, dtype=int)
        image1 = resize(image1, new_size1)
        depth1 = resize(depth1, new_size1)
        resize_factor1 = new_size1/original1_shape[:2]
        
        depth_path2 = pair_metadata['depth_path2']
        
        with h5py.File(depth_path2, 'r') as hdf5_file:
            depth2 = np.array(hdf5_file['/depth'])
        assert(np.min(depth2) >= 0)
        image_path2 = pair_metadata['image_path2']
        
        image2 = io.imread(image_path2, as_gray=True)
        assert(image2.shape[0] == depth2.shape[0] and image2.shape[1] == depth2.shape[1])
        intrinsics2 = pair_metadata['intrinsics2']
        pose2 = pair_metadata['pose2']
        #resize so shortest dimension is fixed in image_size
        original2_shape = image2.shape
        new_size2 = np.asarray(image2.shape[:2]/np.min(image2.shape[:2])*self.image_size, dtype=int)

        image2 = resize(image2, new_size2)
        depth2 = resize(depth2, new_size2)
        resize_factor2 = new_size2/original2_shape[:2]
        
        central_match = pair_metadata['central_match']
        central_match = [central_match[0]*resize_factor1[0], 
                         central_match[1]*resize_factor1[1], 
                         central_match[2]*resize_factor2[0], 
                         central_match[3]*resize_factor2[1]]

        image1, bbox1, image2, bbox2 = self.crop(image1, image2, central_match)
        depth1 = depth1[bbox1[0]:bbox1[0] + self.image_size,
                        bbox1[1]:bbox1[1] + self.image_size]
        depth2 = depth2[bbox2[0]:bbox2[0] + self.image_size,
                        bbox2[1]:bbox2[1] + self.image_size]
        
        resize_matrix1 = [[resize_factor1[0], 0, 0],
                          [0, resize_factor1[1], 0],
                          [0, 0, 1]]  
        intrinsics1 = np.matmul(resize_matrix1, intrinsics1)

        resize_matrix2 = [[resize_factor2[0], 0, 0],
                          [0, resize_factor2[1], 0],
                          [0, 0, 1]]  
        intrinsics2 = np.matmul(resize_matrix2, intrinsics2)
        return (image1, depth1, intrinsics1, pose1, bbox1,
                image2, depth2, intrinsics2, pose2, bbox2)

    def crop(self, image1, image2, central_match):
        bbox1_i = max(int(central_match[0]) - self.image_size // 2, 0)
        if bbox1_i + self.image_size >= image1.shape[0]:
            bbox1_i = image1.shape[0] - self.image_size
        bbox1_j = max(int(central_match[1]) - self.image_size // 2, 0)
        if bbox1_j + self.image_size >= image1.shape[1]:
            bbox1_j = image1.shape[1] - self.image_size

        bbox2_i = max(int(central_match[2]) - self.image_size // 2, 0)
        if bbox2_i + self.image_size >= image2.shape[0]:
            bbox2_i = image2.shape[0] - self.image_size
        bbox2_j = max(int(central_match[3]) - self.image_size // 2, 0)
        if bbox2_j + self.image_size >= image2.shape[1]:
            bbox2_j = image2.shape[1] - self.image_size

        return (image1[bbox1_i : bbox1_i + self.image_size,
                       bbox1_j : bbox1_j + self.image_size],
                       np.array([bbox1_i, bbox1_j]),
                image2[bbox2_i : bbox2_i + self.image_size,
                       bbox2_j : bbox2_j + self.image_size],
                np.array([bbox2_i, bbox2_j]))

    def __getitem__(self, idx):
        (image1, depth1, intrinsics1, pose1, bbox1,
        image2, depth2, intrinsics2, pose2, bbox2) = self.recover_pair(self.dataset[idx])

        image1 = preprocess_image(image1)
        image2 = preprocess_image(image2)

        return {'image1': torch.from_numpy(image1.astype(np.float32)),
                'depth1': torch.from_numpy(depth1.astype(np.float32)),
                'intrinsics1': torch.from_numpy(intrinsics1.astype(np.float32)),
                'pose1': torch.from_numpy(pose1.astype(np.float32)),
                'bbox1': torch.from_numpy(bbox1.astype(np.float32)),
                'image2': torch.from_numpy(image2.astype(np.float32)),
                'depth2': torch.from_numpy(depth2.astype(np.float32)),
                'intrinsics2': torch.from_numpy(intrinsics2.astype(np.float32)),
                'pose2': torch.from_numpy(pose2.astype(np.float32)),
                'bbox2': torch.from_numpy(bbox2.astype(np.float32))}


if __name__ == '__main__':
    from skimage import io
    from torch.utils.data import DataLoader

    root_path = '/media/hudson/9708e369-632b-44b6-8c81-cc636dfdf2f34/home/hudson/Desktop/Unicamp/Doutorado/Projeto/datasets/MegaDepth_v1/'
    train_dset = MegaDepthDataset(scene_list_path='megadepth_utils/train_scenes.txt',
                                    scene_info_path=root_path+'scene_info/',
                                    base_path=root_path,
                                    save_dataset_path='dataset.pkl',
                                    preprocessing='torch',
                                    min_overlap_ratio=0.1,
                                    max_overlap_ratio=0.7,
                                    image_size=720,
                                    train=False)
    train_dataloader = DataLoader(train_dset, batch_size=1, num_workers=1)
    train_dset.build_dataset()
    train_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for batch_idx, batch in train_pbar:
        img0 = batch['image1'][0][0].cpu().numpy()
        depth0 = batch['depth1'][0][0].cpu().numpy()
        depth0 = np.expand_dims(depth0, axis=1)
        print(img0.shape)
        print(depth0.shape)
        io.imshow(img0)
        io.show()
        io.imshow(depth0)
        io.show()
        
        
