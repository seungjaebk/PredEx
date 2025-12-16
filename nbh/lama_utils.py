"""
LAMA Utils Module

LAMA model utilities for map prediction and padding.
"""

import numpy as np
import cv2
import albumentations as A


def get_lama_pred_from_obs(cur_obs_img, lama_model, lama_map_transform, device, 
                           convert_obsimg_to_model_input, visualize_prediction):
    """Get LAMA model prediction from observation image."""
    cur_obs_img_3chan = np.stack([cur_obs_img, cur_obs_img, cur_obs_img], axis=2)
    input_lama_batch, lama_mask = convert_obsimg_to_model_input(cur_obs_img_3chan, lama_map_transform, device)
    lama_pred_alltrain = lama_model(input_lama_batch)
    lama_pred_alltrain_viz = visualize_prediction(lama_pred_alltrain, lama_mask)
    return cur_obs_img_3chan, input_lama_batch, lama_mask, lama_pred_alltrain, lama_pred_alltrain_viz


def get_pred_maputils_from_viz(viz_map):
    """Convert visualization map to prediction map utilities."""
    pred_maputils = np.zeros((viz_map.shape[0], viz_map.shape[1]))
    pred_maputils[viz_map[:,:,0] > 128] = 1  # occ
    return pred_maputils


def get_lama_padding_transform():
    """Get padding transform for LAMA model input."""
    lama_padding_transform = A.PadIfNeeded(
        min_height=None, min_width=None, 
        pad_height_divisor=16, pad_width_divisor=16, 
        border_mode=cv2.BORDER_CONSTANT, value=0
    )
    return lama_padding_transform


def get_padded_obs_map(obs_map):
    """Pad observation map to be divisible by 16."""
    lama_padding_transform = get_lama_padding_transform()
    padded_obs_map = lama_padding_transform(image=obs_map)['image']
    return padded_obs_map


def get_padded_gt_map(gt_map):
    """Pad ground truth map to be divisible by 16."""
    lama_padding_transform = get_lama_padding_transform()
    padded_gt_map = lama_padding_transform(image=gt_map)['image']
    return padded_gt_map
