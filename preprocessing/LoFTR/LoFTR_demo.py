import os

from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt  # 需要导入这个模块来显示图像
from matplotlib.patches import ConnectionPatch
import glob
from tqdm import tqdm

import sys
sys.path.append(r"/root/LoFTR")

from src.utils.plotting import make_matching_figure

from src.loftr import LoFTR, default_cfg

# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.

def run(imfile1,imfile2,out_dir,viz_path,matcher):
    img_name0 = os.path.basename(imfile1)
    img_name1 = os.path.basename(imfile2)

    img0_raw = cv2.imread(imfile1, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(imfile2, cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//8*8, img0_raw.shape[0]//8*8))  # input size shuold be divisible by 8
    img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))
    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    
    matches_path = os.path.join(out_dir,'{}_{}.npz').format(img_name0,img_name1)
    save_path = os.path.join(viz_path,'{}_{}.png').format(img_name0,img_name1)

    np.savez(matches_path, mkpts0=mkpts0, mkpts1=mkpts1, mconf=mconf)



def main(data_dir,out_dir,viz_path):
    default_cfg['coarse']
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load("LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()

    img_files = sorted(glob.glob(os.path.join(data_dir, 'color', '*')))
    num_imgs = len(img_files)
    pbar = tqdm(total=num_imgs * (num_imgs - 1))
    for i in range(num_imgs - 1):
        for j in range(i + 1, num_imgs):
            imfile1 = img_files[i]
            imfile2 = img_files[j]
            run(imfile1,imfile2,out_dir,viz_path,matcher)
            pbar.update(1)

    for i in range(num_imgs - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            imfile1 = img_files[i]
            imfile2 = img_files[j]
            run(imfile1,imfile2,out_dir,viz_path,matcher)
            pbar.update(1)
    pbar.close()
    print('computing all pairwise optical flows for {} is done \n'.format(data_dir))


if __name__ == '__main__':
    data_name = r'case6_9'
    data_dir = r'autodl-tmp/{}'.format(data_name)
    out_dir = r'autodl-tmp/{}/LoFTR_exhaustive'.format(data_name)
    viz_path = r'autodl-tmp/{}/LoFTR_vis'.format(data_name)
    os.makedirs(out_dir, exist_ok=True)
    #os.makedirs(viz_path, exist_ok=True)
    main(data_dir,out_dir,viz_path)