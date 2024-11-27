import os

from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt  # 需要导入这个模块来显示图像
from matplotlib.patches import ConnectionPatch
import sys
sys.path.append(r"D:\Study\Mres\SURG70007_Individual_Project\Code\Code\long_term_correspondence\LoFTR")

from src.utils.plotting import make_matching_figure

from src.loftr import LoFTR, default_cfg

def make_matching_figure0(img0, img1, mkpts0, mkpts1, color, text=[], figsize=(10, 5)):
    """
    绘制匹配结果的可视化图像，包括两个图像及其匹配点的连线。
    
    参数:
        img0: 第一个图像 (numpy array)
        img1: 第二个图像 (numpy array)
        mkpts0: 在第一个图像上的匹配点 (numpy array of shape Nx2)
        mkpts1: 在第二个图像上的匹配点 (numpy array of shape Nx2)
        color: 每个匹配点的颜色 (numpy array of shape Nx4, RGBA)
        text: 在图像上显示的文本信息 (list of strings)
        figsize: 图像大小 (tuple of width, height)
    """
    # 创建一个新的图像，包含两个子图
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

    # 在两个子图上分别显示图像
    ax0.imshow(img0, cmap='gray')
    ax0.axis('off')
    ax1.imshow(img1, cmap='gray')
    ax1.axis('off')

    # 在两个图像之间画线，表示匹配点
    for (pt0, pt1, c) in zip(mkpts0, mkpts1, color):
        # 在第一个图像上的点
        ax0.plot(pt0[0], pt0[1], 'o', markersize=5, markerfacecolor=c, markeredgecolor='white')
        # 在第二个图像上的点
        ax1.plot(pt1[0], pt1[1], 'o', markersize=5, markerfacecolor=c, markeredgecolor='white')
        
        # 连接两个图像的匹配点
        con = ConnectionPatch(xyA=pt1, xyB=pt0, coordsA="data", coordsB="data",
                              axesA=ax1, axesB=ax0, color=c, lw=1)
        ax1.add_artist(con)

    # 在左上角显示文本信息
    for i, t in enumerate(text):
        ax0.text(0, i * 20, t, fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout()
    return fig


# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

default_cfg['coarse']

# Load example images
img0_pth = r"D:\Study\Mres\SURG70007_Individual_Project\Code\Code\omnimotion_with_mask\data\case6_5\color\00000.png"
mask0 = cv2.imread(r'D:\Study\Mres\SURG70007_Individual_Project\Code\Code\omnimotion_with_mask\data\case6_5\\mask\00000.png', cv2.IMREAD_GRAYSCALE)
img1_pth = r"D:\Study\Mres\SURG70007_Individual_Project\Code\Code\omnimotion_with_mask\data\case6_5\color\00050.png"
mask1 = cv2.imread(r'D:\Study\Mres\SURG70007_Individual_Project\Code\Code\omnimotion_with_mask\data\case6_5\mask\00050.png', cv2.IMREAD_GRAYSCALE)
img0_bgr = cv2.imread(img0_pth)
img1_bgr = cv2.imread(img1_pth)
img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//8*8, img0_raw.shape[0]//8*8))  # input size shuold be divisible by 8
img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))

mask0 = cv2.resize(mask0, (mask0.shape[1]//8*8, mask0.shape[0]//8*8))
mask1 = cv2.resize(mask1, (mask1.shape[1]//8*8, mask1.shape[0]//8*8))


img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

# Inference with LoFTR and get prediction
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()


mask0 = mask0.astype(bool)
mask1 = mask1.astype(bool)
                     
valid_matches = []
# Check each match to see if both points are inside the mask
for i, (pt0, pt1) in enumerate(zip(mkpts0, mkpts1)):
    if mask0[int(pt0[1]), int(pt0[0])] and mask1[int(pt1[1]), int(pt1[0])]:
        valid_matches.append(i)

# Use the valid matches to filter the matched keypoints and confidence
'''
mkpts0 = mkpts0[valid_matches]
mkpts1 = mkpts1[valid_matches]
mconf = mconf[valid_matches]
'''
mconf0 = mconf > 0.2
mkpts0 = mkpts0[mconf0]
mkpts1 = mkpts1[mconf0]
mconf = mconf[mconf0]


# Draw
color = cm.jet(mconf)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]
fig = make_matching_figure(img0_rgb, img1_rgb, mkpts0, mkpts1, color, text=text)

# 显示图像
plt.show()

# 保存fig为图像文件
output_path = r"D:\Study\Mres\SURG70007_Individual_Project\dissertation\dissertation_vis\LoFTR.png"
fig.savefig(output_path, dpi=300, bbox_inches='tight')