import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def resize_logits(logits, size):
    if tuple(logits.shape[-2:]) == tuple(size):
        return logits
    return F.interpolate(logits, size=size, mode='bilinear', align_corners=True)


def logits_to_mask(logits):
    if logits.shape[1] == 1:
        probs = logits.sigmoid()
        return (probs >= 0.5).long().squeeze(1)
    return torch.argmax(logits, dim=1)


def batch_confusion_matrix(pred_mask, gt_mask):
    pred_mask = pred_mask.long()
    gt_mask = gt_mask.long()

    tp = torch.logical_and(pred_mask == 1, gt_mask == 1).sum().item()
    fp = torch.logical_and(pred_mask == 1, gt_mask == 0).sum().item()
    fn = torch.logical_and(pred_mask == 0, gt_mask == 1).sum().item()
    tn = torch.logical_and(pred_mask == 0, gt_mask == 0).sum().item()
    return tp, fp, fn, tn


def aggregate_binary_metrics(tp, fp, fn, tn):
    eps = 1e-6
    fg_iou = tp / (tp + fp + fn + eps)
    fg_dice = (2 * tp) / (2 * tp + fp + fn + eps)
    fg_recall = tp / (tp + fn + eps)

    bg_iou = tn / (tn + fp + fn + eps)
    bg_recall = tn / (tn + fp + eps)

    return {
        'IoU': 100.0 * fg_iou,
        'Dice': 100.0 * fg_dice,
        'Recall': 100.0 * fg_recall,
        'mIoU': 100.0 * ((fg_iou + bg_iou) / 2.0),
        'mACC': 100.0 * ((fg_recall + bg_recall) / 2.0),
    }


def format_metrics(metrics):
    return '\n'.join([
        '    IoU = {:.2f}'.format(metrics['IoU']),
        '    Dice = {:.2f}'.format(metrics['Dice']),
        '    Recall = {:.2f}'.format(metrics['Recall']),
        '    mIoU = {:.2f}'.format(metrics['mIoU']),
        '    mACC = {:.2f}'.format(metrics['mACC']),
    ])


def save_prediction_mask(pred_mask, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    pred = pred_mask.detach().cpu().numpy().astype(np.uint8) * 255
    Image.fromarray(pred, mode='L').save(destination)
