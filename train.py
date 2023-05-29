import os
import cv2
import torch
import argparse
import multiprocessing
import numpy as np
from tqdm import tqdm
from skimage import io
import matplotlib.cm as cm
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from dataset import MegaDepthDataset
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from loss import nll_loss
from ground_truth import get_ground_truth
from utils import pad_data, load_model, load_model_weights, save_model, scores_to_matches, \
                  ohe_to_le, create_kpts_image, create_matches_image
    




def train(lr, num_epochs, save_every, pos_weight, neg_weight, train_dataloader, validation_dataloader, 
         load_model_path, max_iter, checkpoints_path, config, device, writer, 
         only_val=False):
    
    iter = 0
    superpoint = SuperPoint(config.get('superpoint', {})).to(device)

    superglue = SuperGlue(config.get('superglue', {})).to(device)
    optimizer = optim.Adam(superglue.parameters(), lr=lr)
    start_epoch = 1
    start_step = 0
    loss = None
    if(load_model_path == None):
        path = Path(__file__).parent
        path = path / 'models/weights/superglue_outdoor.pth'
        superglue = load_model(superglue, path)
    else:
        superglue, start_epoch, start_step, optimizer_state, loss = load_model_weights(superglue, load_model_path, 
                                                                                       recover_state=True,
                                                                                       modules=['kenc', 'gnn', 'final_proj'])
        print('starting from epoch ', start_epoch)
        print('starting from step ', start_step)
        optimizer.load_state_dict(optimizer_state)

    for epoch_idx in range(start_epoch, num_epochs + 1):
        train_size = min(max_iter, len(train_dataloader))
        train_pbar = tqdm(enumerate(train_dataloader), total=train_size)
        if(validation_dataloader != None):
            val_size = min(max_iter, len(validation_dataloader))
            val_pbar = tqdm(enumerate(validation_dataloader), total=val_size)
        training_losses = []
        validation_losses = []
                        
        if(not only_val):
            print('\n')
            print('='*20)
            print('Training...')
            if(epoch_idx == start_epoch and start_step > max_iter):
                continue
            
            for batch_idx, batch in train_pbar:
                if(batch_idx >= max_iter):
                    break
                if(batch_idx < start_step and epoch_idx == start_epoch):
                    continue
                superglue.train()
                optimizer.zero_grad()
                img0 = batch['image1'].to(device)                
                img1 = batch['image2'].to(device)
                
                kpts = {}
                sp1 = superpoint({'image': img0})
                kpts = {**kpts, **{k+'0': v for k, v in sp1.items()}}
                sp2 = superpoint({'image': img1})
                kpts = {**kpts, **{k+'1': v for k, v in sp2.items()}}
                data = {'image0': img0, 'image1': img1}
                data = {**data, **kpts}
                
                data = pad_data(data, config['superpoint']['max_keypoints'], 
                                img0.shape, device)

                gt_matches = get_ground_truth(data['keypoints0'], 
                                              data['keypoints1'], 
                                              batch, device)
                if(gt_matches == None):
                    continue
                
                #Forward
                matches = superglue(data)['scores_matrix']
                #LOSS
                loss = nll_loss(matches, gt_matches, pos_weight=pos_weight, 
                                neg_weight=neg_weight)
                if(loss != None):
                    loss.backward()
                    optimizer.step()
                    current_loss = loss.item()
                    training_losses.append(current_loss)
                    train_pbar.set_postfix(loss=('%.4f' % np.mean(training_losses)))
                    
                    if(batch_idx%save_every == 0):
                        output_name = f'model_{epoch_idx}_{batch_idx}'
                        save_model(os.path.join(checkpoints_path, output_name+".pth"), 
                                superglue, optimizer, batch_idx, epoch_idx, loss)
                        #title = 'Training_loss_iterations '
                        #writer.add_scalar(title, np.mean(training_losses), iter)
                        #writer.flush()
                    iter+=1

            #Adding predicted matches to tensorboard
            m_tb, sc_tb = scores_to_matches(matches, config['superglue']['match_threshold'])
            m_tb = m_tb.detach().cpu().numpy()
            sc_tb = sc_tb.detach().cpu().numpy()
            imgs0_cpu = [im0.cpu().numpy() * 255 for im0 in data['image0']]
            imgs1_cpu = [im1.cpu().numpy() * 255 for im1 in data['image1']]
            kpts0_cpu = [k0.cpu().numpy() for k0 in data['keypoints0']]
            kpts1_cpu = [k1.cpu().numpy() for k1 in data['keypoints1']]
            m_imgs = np.array([create_matches_image(im0[0], im1[0], k0, k1, m, s)
                        for im0, im1, k0, k1, m, s 
                        in zip(imgs0_cpu, imgs1_cpu, kpts0_cpu, kpts1_cpu, m_tb, sc_tb)])
            m_imgs = torch.from_numpy(m_imgs).permute(0, 3, 1, 2)
            imgs_grid = vutils.make_grid(m_imgs)
            writer.add_image('train/pred', imgs_grid, epoch_idx)
            
            #Adding ground truth matches to tensorboard
            gt_m_tb, gt_sc_tb = scores_to_matches(gt_matches, config['superglue']['match_threshold'])
            gt_m_tb = gt_m_tb.detach().cpu().numpy()
            gt_sc_tb = gt_sc_tb.detach().cpu().numpy()
            gt_m_imgs = np.array([create_matches_image(im0[0], im1[0], k0, k1, m, s)
                            for im0, im1, k0, k1, m, s 
                            in zip(imgs0_cpu, imgs1_cpu, kpts0_cpu, kpts1_cpu, gt_m_tb, gt_sc_tb)])

            gt_m_imgs = torch.from_numpy(gt_m_imgs).permute(0, 3, 1, 2)
            gt_imgs_grid = vutils.make_grid(gt_m_imgs)
            writer.add_image('train/gt', gt_imgs_grid, epoch_idx)
            writer.add_scalar('Training_loss ', np.mean(training_losses), epoch_idx)
            writer.flush()
            
        if(validation_dataloader != None):
            print('\n')
            print('='*20)
            print('Validation...')
            #Validation Loop
            superglue.eval()
            with torch.no_grad():
                for batch_idx, batch in val_pbar:
                    if(batch_idx >= max_iter):
                        break
                    img0 = batch['image1'].to(device)
                    img1 = batch['image2'].to(device)

                    kpts = {}
                    sp1 = superpoint({'image': img0})
                    kpts = {**kpts, **{k+'0': v for k, v in sp1.items()}}
                    sp2 = superpoint({'image': img1})
                    kpts = {**kpts, **{k+'1': v for k, v in sp2.items()}}
                    data = {'image0': img0, 'image1': img1}
                    data = {**data, **kpts}
                    data = pad_data(data, config['superpoint']['max_keypoints'],
                                    img0.shape, device)
                    gt_matches = get_ground_truth(data['keypoints0'], 
                                        data['keypoints1'], 
                                        batch, device)
                    #Forward
                    matches = superglue(data)['scores_matrix']
                    
                    #LOSS
                    loss = nll_loss(matches, gt_matches, pos_weight=pos_weight, 
                                    neg_weight=neg_weight)
                    current_loss = loss.item()
                    validation_losses.append(current_loss)
                    val_pbar.set_postfix(loss=('%.4f' % np.mean(validation_losses)))                                   

                #Adding predicted matches to tensorboard
                m_tb, sc_tb = scores_to_matches(matches, config['superglue']['match_threshold'])
                m_tb = m_tb.detach().cpu().numpy()
                sc_tb = sc_tb.detach().cpu().numpy()
                imgs0_cpu = [im0.cpu().numpy() * 255 for im0 in data['image0']]
                imgs1_cpu = [im1.cpu().numpy() * 255 for im1 in data['image1']]
                kpts0_cpu = [k0.cpu().numpy() for k0 in data['keypoints0']]
                kpts1_cpu = [k1.cpu().numpy() for k1 in data['keypoints1']]
                m_imgs = np.array([create_matches_image(im0[0], im1[0], k0, k1, m, s)
                            for im0, im1, k0, k1, m, s 
                            in zip(imgs0_cpu, imgs1_cpu, kpts0_cpu, kpts1_cpu, m_tb, sc_tb)])
                m_imgs = torch.from_numpy(m_imgs).permute(0, 3, 1, 2)
                imgs_grid = vutils.make_grid(m_imgs)
                writer.add_image('val/pred', imgs_grid, epoch_idx)
            
                #Adding ground truth matches to tensorboard
                gt_m_tb, gt_sc_tb = scores_to_matches(gt_matches, config['superglue']['match_threshold'])
                gt_m_tb = gt_m_tb.detach().cpu().numpy()
                gt_sc_tb = gt_sc_tb.detach().cpu().numpy()
                gt_m_imgs = np.array([create_matches_image(im0[0], im1[0], k0, k1, m, s)
                                for im0, im1, k0, k1, m, s 
                                in zip(imgs0_cpu, imgs1_cpu, kpts0_cpu, kpts1_cpu, gt_m_tb, gt_sc_tb)])

                gt_m_imgs = torch.from_numpy(gt_m_imgs).permute(0, 3, 1, 2)
                gt_imgs_grid = vutils.make_grid(gt_m_imgs)
                writer.add_image('val/gt', gt_imgs_grid, epoch_idx)

            title = 'Validation_loss '
            writer.add_scalar(title, np.mean(validation_losses), epoch_idx)
            writer.flush()
        
        if(not only_val):
            output_name = f'model_{epoch_idx}'
            save_model(os.path.join(checkpoints_path, output_name+".pth"), 
                    superglue, optimizer, len(train_dataloader), epoch_idx, loss)
    


def main(lr, batch_size, num_epochs, save_every, dataset_path, train_scenes_path, 
        load_model_path, valid_scenes_path, logs_dir, max_iter,
        checkpoints_path, save_dataset_path, load_dataset_from_file,
        pos_weight, neg_weight, only_val):
    
    MAX_KPTS = 400    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    config = {'superpoint': {'nms_radius': 4,
                            'keypoint_threshold': 0.005,
                            'max_keypoints': MAX_KPTS},
              'superglue': {'weights': 'outdoor',
                            'sinkhorn_iterations': 30,
                            'match_threshold': 0.2}}

    scenes_info_path = os.path.join(dataset_path, 'scene_info')
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")                              

    random_seed = 33
    #Seed
    torch.manual_seed(random_seed)
    if use_cuda:
        torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    train_dset = MegaDepthDataset(scene_list_path=train_scenes_path,
                            scene_info_path=scenes_info_path,
                            base_path=dataset_path,
                            preprocessing='torch',
                            min_overlap_ratio=0.1,
                            max_overlap_ratio=0.7,
                            image_size=720,
                            save_dataset_path=os.path.join(save_dataset_path, "train_dset.pkl"),
                            load_from_file=load_dataset_from_file)

    val_dset = MegaDepthDataset(scene_list_path=valid_scenes_path,
                                scene_info_path=scenes_info_path,
                                base_path=dataset_path,
                                train=False,
                                preprocessing='torch',
                                min_overlap_ratio=0.1,
                                max_overlap_ratio=0.7,
                                image_size=720,
                                save_dataset_path=os.path.join(save_dataset_path, "valid_dset.pkl"),
                                load_from_file=load_dataset_from_file)

    train_dataloader = DataLoader(train_dset, batch_size=batch_size, num_workers=multiprocessing.cpu_count())
    train_dset.build_dataset()
    
    validation_dataloader = DataLoader(val_dset, batch_size=batch_size, num_workers=multiprocessing.cpu_count())
    val_dset.build_dataset()

    writer = SummaryWriter(logs_dir,
        comment= "_LR_"+ str(lr) + "_Batch_size_" + str(batch_size))
    
    train(lr, num_epochs, save_every, pos_weight, neg_weight, 
          train_dataloader, validation_dataloader, load_model_path,
          max_iter, checkpoints_path, config, device, writer, 
          only_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", default=None, help="Path to dataset")
    parser.add_argument("train_scenes_path", default=None, help="Path to train scenes txt")
    parser.add_argument("valid_scenes_path", default=None, help="Path to valid scenes txt")
    parser.add_argument("--load_model_path", default=None, help="Path to load model")
    parser.add_argument("--logs_dir", default="logs/", help="Path to save logs")
    parser.add_argument("--max_iter", default=1000, type=int, help="Max training iterations")
    parser.add_argument("--checkpoints_path", default="models/", help="Path to save checkpoints")   
    parser.add_argument("--save_dataset_path", default="logs/", help="Path to save built dataset")   
    parser.add_argument("--load_dataset_from_file", action='store_true', help="True if we should load the dataset from a pickle file")   
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning Rate")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch Size")
    parser.add_argument("--num_epochs", default=100, type=int, help="Path to logs")
    parser.add_argument("--save_every", default=200, type=int, help="Save model after this number of iterations")
    parser.add_argument("--pos_weight", default=0.5, type=float, help="Weight to compute loss in positive samples")
    parser.add_argument("--neg_weight", default=0.5, type=float, help="Weight to compute loss in negative samples")
    parser.add_argument("--only_val", action='store_true')

    args = parser.parse_args()
    
    main(args.learning_rate, args.batch_size, args.num_epochs, args.save_every,
         args.dataset_path, args.train_scenes_path, args.load_model_path, 
         args.valid_scenes_path, args.logs_dir, args.max_iter, args.checkpoints_path, 
         args.save_dataset_path, args.load_dataset_from_file,
         args.pos_weight, args.neg_weight, args.only_val)