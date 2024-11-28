import os
import numpy as np
import csv
from config import config_parser
import torch
from trainer import BaseTrainer
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import imageio
import pickle

def eval_temporal_coherence(pred_tracks, gt_tracks, gt_occluded):
    '''
    :param pred_tracks: [1, n_pts, n_imgs, 2]
    :param gt_tracks: [1, n_pts, n_imgs, 2]
    :param gt_occluded: [1, n_pts, n_imgs] bool
    :return:
    '''
    pred_flow_01 = pred_tracks[..., 1:-1, :] - pred_tracks[..., :-2, :]
    pred_flow_12 = pred_tracks[..., 2:, :] - pred_tracks[..., 1:-1, :]
    gt_occluded_3 = gt_occluded[..., :-2] | gt_occluded[..., 1:-1] | gt_occluded[..., 2:]
    gt_visible_3 = ~gt_occluded_3

    # difference in acceleration
    gt_flow_01 = gt_tracks[..., 1:-1, :] - gt_tracks[..., :-2, :]
    gt_flow_12 = gt_tracks[..., 2:, :] - gt_tracks[..., 1:-1, :]
    flow_diff = np.linalg.norm(pred_flow_12 - pred_flow_01 - (gt_flow_12 - gt_flow_01), axis=-1)
    error = flow_diff[gt_visible_3].sum() / gt_visible_3.sum()
    return error


def eval_one_sequence(args, annotation_dir, dataset_type, seq_name, out_dir, occlusion_th=0.99):
    torch.manual_seed(1234)  # need to keep the same seed as training otherwise will lead to issues when doing things like for-loops 
    model = BaseTrainer(args)

    os.makedirs(out_dir, exist_ok=True)
    annotation_file = '{}/{}.pkl'.format(annotation_dir, seq_name)
    print('evaluating {}...'.format(seq_name))

    use_max_loc = True

    # Load tapvid data
    inputs = np.load(annotation_file, allow_pickle=True)

    video = inputs['video']
    query_points = inputs[dataset_type]['query_points']
    target_points = inputs[dataset_type]['target_points']
    gt_occluded = inputs[dataset_type]['occluded']
    gt_trackgroup = inputs[dataset_type]['trackgroup']

    one_hot_eye = np.eye(target_points.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0 #


    ids1 = query_points[0, :, 0].astype(int)
    px1s = torch.from_numpy(query_points[:, :, [2, 1]]).transpose(0, 1).float().cuda()

    results = np.zeros(target_points.shape)
    occlusions = np.zeros(gt_occluded.shape)
    with torch.no_grad():
        for i in range(gt_occluded.shape[-1]):
            results_, occlusions_ = model.get_correspondences_and_occlusion_masks_for_pixels(
                ids1=ids1, 
                px1s=px1s, ids2=[i for _ in range(px1s.shape[0])],
                use_max_loc=use_max_loc)
            results[:, :, i, :] = results_.transpose(0, 1).cpu().numpy()
            occlusions[:, :, i] = occlusions_.squeeze().cpu().numpy()

    
    target_points /= (model.w / 256, model.h / 256)
    results /= (model.w / 256, model.h / 256)

    metrics = {}
    #true indicate occluded
    occlusion_mask = occlusions > occlusion_th
    out_of_boundary_mask = (results[..., 0] < 0) | (results[..., 0] > model.w - 1) | \
                            (results[..., 1] < 0) | (results[..., 1] > model.h - 1)
    out_of_boundary_mask2 = (target_points[..., 0] < 0) & (target_points[..., 1] < 0)
    occlusion_mask = occlusion_mask | out_of_boundary_mask
    evaluation_points = evaluation_points & ~out_of_boundary_mask2
    occ_acc = np.sum(np.equal(occlusion_mask, gt_occluded) & evaluation_points, axis=(1, 2)) / np.sum(evaluation_points)
    metrics['occlusion_accuracy'] = occ_acc

    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(occlusion_mask)
    all_frac_within = []
    all_jaccard = []

    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum(
            np.square(results - target_points),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(
            visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics['pts_within_' + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2))
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics['jaccard_' + str(thresh)] = jaccard
        all_jaccard.append(jaccard)

    metrics['average_pts_within_thresh'] = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    metrics['average_jaccard'] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    metrics['temporal_coherence'] = eval_temporal_coherence(results, target_points, gt_occluded)

    metrics = dict(sorted(metrics.items()))
    with open(os.path.join(out_dir, '{}.csv'.format(seq_name)),
                'w', newline='') as csvfile:
        fieldnames = ['video_name',
                        'average_jaccard',
                        'average_pts_within_thresh',
                        'occlusion_accuracy',
                        'temporal_coherence',
                        'jaccard_1',
                        'jaccard_2',
                        'jaccard_4',
                        'jaccard_8',
                        'jaccard_16',
                        'pts_within_1',
                        'pts_within_2',
                        'pts_within_4',
                        'pts_within_8',
                        'pts_within_16',
                        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'video_name': seq_name,
            'average_jaccard': metrics['average_jaccard'].item(),
            'average_pts_within_thresh': metrics['average_pts_within_thresh'].item(),
            'occlusion_accuracy': metrics['occlusion_accuracy'].item(),
            'temporal_coherence': metrics['temporal_coherence'].item(),
            'jaccard_1': metrics['jaccard_1'].item(),
            'jaccard_2': metrics['jaccard_2'].item(),
            'jaccard_4': metrics['jaccard_4'].item(),
            'jaccard_8': metrics['jaccard_8'].item(),
            'jaccard_16': metrics['jaccard_16'].item(),
            'pts_within_1': metrics['pts_within_1'].item(),
            'pts_within_2': metrics['pts_within_2'].item(),
            'pts_within_4': metrics['pts_within_4'].item(),
            'pts_within_8': metrics['pts_within_8'].item(),
            'pts_within_16': metrics['pts_within_16'].item()
        })
        # del model
        # torch.cuda.empty_cache()


def summarize(out_dir, dataset_type):
    result_files = sorted(glob.glob(os.path.join(out_dir, '*.csv')))
    sum_file = os.path.join(out_dir, '{}.csv'.format(dataset_type))
    
    flag = True
    num_seqs = 0
    with open(sum_file, 'w', newline='') as outfile:
        for i, result_file in enumerate(result_files):
            with open(result_file, 'r', newline='') as infile:
                reader = csv.DictReader(infile)
                if flag:
                    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    flag = False
                for row in reader:
                    writer.writerow(row)
                num_seqs += 1

    df = pd.read_csv(sum_file)
    average = {'video_name': 'average'}
    for k in df.keys()[1:]:
        average[k] = np.round(df[k].mean(), 5)
        df[k] = np.round(df[k], 5)

    df.loc[-1] = average
    df.index = df.index + 1
    df.sort_index(inplace=True)
    df.to_csv(sum_file, index=False)
    print('{} | average_jaccard: {:.5f} | average_pts_within_thresh: {:.5f} '
            '| occlusion_acc: {:.5f} | temporal_coherence: {:.5f}'.
            format(num_seqs,
                   average['average_jaccard'],
                   average['average_pts_within_thresh'],
                   average['occlusion_accuracy'],
                   average['temporal_coherence']
                   ))


if __name__ == '__main__':
    args = config_parser()

    ckpt_dir = args.save_dir
    annotation_dir = r'.\dataset\annotation'
    seq_names = os.listdir(ckpt_dir)
    dataset_type = 'tools' # tools or tissue
    out_dir = r'\Evaluation\{}'.format(dataset_type)
    os.makedirs(out_dir, exist_ok=True)
    
    
    for seq_name in seq_names:
        args.ckpt_path = os.path.join(ckpt_dir, seq_name, 'model_100000.pth')
        eval_one_sequence(args, annotation_dir, dataset_type, seq_name, out_dir=out_dir)


    #summarize(out_dir, dataset_type)


