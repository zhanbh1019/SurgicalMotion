import os
import glob
import json
import imageio
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import multiprocessing as mp
from util import normalize_coords, gen_grid_np


def get_sample_weights(flow_stats):
    sample_weights = {}
    for k in flow_stats.keys():
        sample_weights[k] = {}
        total_num = np.array(list(flow_stats[k].values())).sum()
        for j in flow_stats[k].keys():
            sample_weights[k][j] = 1. * flow_stats[k][j] / total_num
    return sample_weights


class RAFTExhaustiveDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        self.mask_dir = os.path.join(self.seq_dir, 'mask')
        self.flow_dir = os.path.join(self.seq_dir, 'raft_exhaustive')
        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.img_names = img_names[:self.num_imgs]

        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        max_interval = self.num_imgs - 1 if not max_interval else max_interval
        self.max_interval = mp.Value('i', max_interval)
        self.num_pts = self.args.num_pts
        self.grid = gen_grid_np(self.h, self.w)
        flow_stats = json.load(open(os.path.join(self.seq_dir, 'flow_stats.json')))
        self.sample_weights = get_sample_weights(flow_stats)

    def __len__(self):
        return self.num_imgs * 100000

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def increase_max_interval_by(self, increment):
        curr_max_interval = self.max_interval.value
        self.max_interval.value = min(curr_max_interval + increment, self.num_imgs - 1)

    def __getitem__(self, idx):
        cached_flow_pred_dir = os.path.join('out', '{}_{}'.format(self.args.expname, self.seq_name), 'flow')
        cached_flow_pred_files = sorted(glob.glob(os.path.join(cached_flow_pred_dir, '*')))
        flow_error_file = os.path.join(os.path.dirname(cached_flow_pred_dir), 'flow_error.txt')
        if os.path.exists(flow_error_file):
            flow_error = np.loadtxt(flow_error_file)
            id1_sample_weights = flow_error / np.sum(flow_error)
            id1 = np.random.choice(self.num_imgs, p=id1_sample_weights)
        else:
            id1 = idx % self.num_imgs

        img_name1 = self.img_names[id1]
        max_interval = min(self.max_interval.value, self.num_imgs - 1)
        img2_candidates = sorted(list(self.sample_weights[img_name1].keys()))
        img2_candidates = img2_candidates[max(id1 - max_interval, 0):min(id1 + max_interval, self.num_imgs - 1)]

        # sample more often from i-1 and i+1
        id2s = np.array([self.img_names.index(n) for n in img2_candidates])
        sample_weights = np.array([self.sample_weights[img_name1][i] for i in img2_candidates])
        sample_weights /= np.sum(sample_weights)
        sample_weights[np.abs(id2s - id1) <= 1] = 0.5
        sample_weights /= np.sum(sample_weights)

        img_name2 = np.random.choice(img2_candidates, p=sample_weights)
        id2 = self.img_names.index(img_name2)
        frame_interval = abs(id1 - id2)

        # read image, flow and confidence
        img1 = imageio.imread(os.path.join(self.img_dir, img_name1)) / 255.
        img2 = imageio.imread(os.path.join(self.img_dir, img_name2)) / 255.

        mask1 = cv2.imread(os.path.join(self.mask_dir, img_name1), cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(os.path.join(self.mask_dir, img_name2), cv2.IMREAD_GRAYSCALE)

        flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(img_name1, img_name2))
        flow = np.load(flow_file)
        mask_file = flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
        masks = imageio.imread(mask_file) / 255.

        LoFTR_path = flow_file.replace('raft_exhaustive', 'LoFTR_exhaustive').replace('.npy', '.npz')
        sparse_points = np.load(LoFTR_path)    #[x,y] 
        sparse_conf = sparse_points['mconf'] > 0.5
        sparse1 = sparse_points['mkpts0'][sparse_conf]
        sparse2 = sparse_points['mkpts1'][sparse_conf]
        
        mask_sparse  = np.ones(masks[..., 0].shape, dtype=bool)
        mask_sparse[sparse1[:, 1].astype(int), sparse1[:, 0].astype(int)] = False 

        selected_sparse_points = np.random.choice(len(sparse1), size=round(self.num_pts/4), replace=(len(sparse1) < round(self.num_pts/4)))  
        #sparse_corr[selected_sparse_points]
        pts1_sparse = torch.from_numpy(sparse1[selected_sparse_points,:]).float()
        pts2_sparse = torch.from_numpy(sparse2[selected_sparse_points,:]).float()


        coord1 = self.grid
        coord2 = self.grid + flow

        cycle_consistency_mask = masks[..., 0] > 0
        occlusion_mask = masks[..., 1] > 0

        if frame_interval == 1:
            mask = np.ones_like(cycle_consistency_mask) & mask_sparse
            flow_mask = np.ones_like(cycle_consistency_mask)
        else:
            #mask = cycle_consistency_mask | occlusion_mask
            mask = np.ones_like(cycle_consistency_mask) & mask_sparse
            flow_mask = cycle_consistency_mask | occlusion_mask
        
        if mask.sum() == 0:
            invalid = True
            mask = np.ones_like(cycle_consistency_mask)
        else:
            invalid = False

        if len(cached_flow_pred_files) > 0 and self.args.use_error_map:
            cached_flow_pred_file = cached_flow_pred_files[id1]
            assert img_name1 + '_' in cached_flow_pred_file
            sup_flow_file = os.path.join(self.flow_dir, os.path.basename(cached_flow_pred_file))
            pred_flow = np.load(cached_flow_pred_file)
            sup_flow = np.load(sup_flow_file)
            error_map = np.linalg.norm(pred_flow - sup_flow, axis=-1)
            error_map = cv2.GaussianBlur(error_map, (5, 5), 0)
            error_selected = error_map[mask]
            prob = error_selected / np.sum(error_selected)
            select_ids_error = np.random.choice(mask.sum(), round(self.num_pts*3/4), replace=(mask.sum() < round(self.num_pts*3/4)), p=prob)
            select_ids_random = np.random.choice(mask.sum(), round(self.num_pts*3/4), replace=(mask.sum() < round(self.num_pts*3/4)))
            select_ids = np.random.choice(np.concatenate([select_ids_error, select_ids_random]), round(self.num_pts*3/4),
                                          replace=False)
        else:
            if self.args.use_count_map:
                count_map = imageio.imread(os.path.join(self.seq_dir, 'count_maps', img_name1.replace('.jpg', '.png')))
                pixel_sample_weight = 1 / np.sqrt(count_map + 1.)
                pixel_sample_weight = pixel_sample_weight[mask]
                pixel_sample_weight /= pixel_sample_weight.sum()
                select_ids = np.random.choice(mask.sum(), round(self.num_pts*3/4), replace=(mask.sum() < round(self.num_pts*3/4)),
                                              p=pixel_sample_weight)
            else:
                select_ids = np.random.choice(mask.sum(), round(self.num_pts*3/4), replace=(mask.sum() < round(self.num_pts*3/4)))

        pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 2)

        pts1 = torch.from_numpy(coord1[mask][select_ids]).float()
        pts2 = torch.from_numpy(coord2[mask][select_ids]).float()

        pts1 = torch.cat((pts1, pts1_sparse), dim=0)
        pts2 = torch.cat((pts2, pts2_sparse), dim=0)

        pts2_normed = normalize_coords(pts2[0:round(self.num_pts*3/4),:], self.h, self.w)[None, None]

        flow_mask = torch.from_numpy(flow_mask[mask][select_ids])
        covisible_mask = torch.from_numpy(cycle_consistency_mask[mask][select_ids]).float()[..., None]
        covisible_mask_sparse =  torch.ones(round(self.num_pts/4), 1)
        covisible_mask = torch.cat((covisible_mask, covisible_mask_sparse),dim=0)

        weights = torch.ones_like(covisible_mask) * pair_weight

        gt_rgb1 = torch.from_numpy(img1[mask][select_ids]).float()
        gt_rgb2 = F.grid_sample(torch.from_numpy(img2).float().permute(2, 0, 1)[None], pts2_normed,
                                align_corners=True).squeeze().T
        
        gt_rgb1_sparse = torch.from_numpy(img1[pts1_sparse[...,1].int(),pts1_sparse[...,0].int()]).float()
        gt_rgb2_sparse = torch.from_numpy(img2[pts2_sparse[...,1].int(),pts2_sparse[...,0].int()]).float()

        gt_rgb1 = torch.cat((gt_rgb1, gt_rgb1_sparse), dim=0)
        gt_rgb2 = torch.cat((gt_rgb2, gt_rgb2_sparse), dim=0)
    

        mask_1 = torch.from_numpy(mask1[mask][select_ids]).float()
        mask_2 = F.grid_sample(torch.from_numpy(mask2)[None][None].float(), pts2_normed,
                                align_corners=True).squeeze().T
        
        mask1_sparse = torch.from_numpy(mask1[pts1_sparse[...,1].int(),pts1_sparse[...,0].int()]).float()
        mask2_sparse = torch.from_numpy(mask2[pts2_sparse[...,1].int(),pts2_sparse[...,0].int()]).float()
        
        mask_1 = torch.cat((mask_1, mask1_sparse), dim=0)
        mask_2 = torch.cat((mask_2, mask2_sparse), dim=0)
        
        if invalid:
            weights = torch.zeros_like(weights)

        if np.random.choice([0, 1]):
            id1, id2, pts1, pts2, gt_rgb1, gt_rgb2, mask_1, mask_2, mask1, mask2   \
                = id2, id1, pts2, pts1, gt_rgb2, gt_rgb1, mask_2, mask_1, mask2, mask1
            weights[covisible_mask == 0.] = 0
        
        data = {'ids1': id1,
                'ids2': id2,
                'pts1': pts1,  # [n_pts, 2]
                'pts2': pts2,  # [n_pts, 2]
                'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                'gt_rgb2': gt_rgb2,
                'mask_1': mask_1,
                'mask_2': mask_2,
                'mask2': mask2,
                'weights': weights,  # [n_pts, 1]
                'covisible_mask': covisible_mask,  # [n_pts, 1]
                'flow_mask': flow_mask,
                }
        return data