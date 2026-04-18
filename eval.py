import os

import torch
import torch.utils.data
from torch import nn
import transforms as T
import utils
from pretrained_assets import prepare_official_pretrained_assets
from segmentation_metrics import aggregate_binary_metrics, batch_confusion_matrix, format_metrics, logits_to_mask, resize_logits, save_prediction_mask

import numpy as np
from PIL import Image
import torch.nn.functional as F


def get_dataset(image_set, transform, args):
    if args.dataset == 'plantseg':
        from data.dataset_plantseg import PlantSegDataset
        ds = PlantSegDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
        return ds, 2

    from data.dataset_refer_bert import ReferDatasetTest
    ds = ReferDatasetTest(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes

def batch_IoU(pred, gt):
    intersection = torch.logical_and(pred, gt).sum(1)
    union = torch.logical_or(pred, gt).sum(1)
    # intersection = torch.sum(torch.mul(pred, gt), dim=1)
    # union = torch.sum(torch.add(pred, gt), dim=1) - intersection

    iou = intersection.float() / union.float()

    return iou, intersection, union

def batch_evaluate_refer(model, data_loader):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_num = len(data_loader.dataset)
    acc_ious = torch.zeros(1).cuda()

    # evaluation variables
    cum_I = torch.zeros(1).cuda()
    cum_U = torch.zeros(1).cuda()
    eval_seg_iou_list = [.5, .7, .9]
    seg_correct = torch.zeros(len(eval_seg_iou_list)).cuda()

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):

            image, targets, sentences, attentions = data
            image, sentences, attentions = image.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)
            target = targets['mask'].cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            output = model(image, sentences, l_mask=attentions)

            iou, I, U = batch_IoU(output.flatten(1), target.flatten(1))
            acc_ious += iou.sum()
            cum_I += I.sum()
            cum_U += U.sum()
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou).sum()

    torch.cuda.synchronize()
    cum_I = cum_I.cpu().numpy()
    cum_U = cum_U.cpu().numpy()
    acc_ious = acc_ious.cpu().numpy()
    seg_correct = seg_correct.cpu().numpy()

    mIoU = acc_ious / total_num
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / total_num)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

def batch_evaluate_plantseg(model, data_loader, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    confusion = torch.zeros(4, device='cuda', dtype=torch.float64)
    pred_mask_root = os.path.join(args.output_dir, args.pred_mask_subdir)
    os.makedirs(pred_mask_root, exist_ok=True)

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, targets, sentences, attentions = data
            image = image.cuda(non_blocking=True)
            sentences = sentences.cuda(non_blocking=True).squeeze(1)
            attentions = attentions.cuda(non_blocking=True).squeeze(1)
            target = targets['mask'].cuda(non_blocking=True).long()

            output = model(image, sentences, l_mask=attentions, resize_output=False, return_probs=True)
            output = resize_logits(output, target.shape[-2:])
            pred_mask = logits_to_mask(output)

            confusion += torch.tensor(batch_confusion_matrix(pred_mask, target), device=confusion.device,
                                      dtype=confusion.dtype)

            for idx, mask_path in enumerate(targets['mask_path']):
                save_prediction_mask(pred_mask[idx], os.path.join(pred_mask_root, mask_path))

    torch.cuda.synchronize()
    tp, fp, fn, tn = confusion.tolist()
    metrics = aggregate_binary_metrics(tp, fp, fn, tn)
    print('Final results:')
    print(format_metrics(metrics))
    with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(format_metrics(metrics) + '\n')
    return metrics

def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size, eval_mode=args.eval_ori_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)

def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg),axis=1)
    U = np.sum(np.logical_or(pred_seg, gd_seg),axis=1)

    return I, U

def batch_evaluate(model, data_loader, args):
    if args.dataset == 'plantseg':
        return batch_evaluate_plantseg(model, data_loader, args)
    return batch_evaluate_refer(model, data_loader)

def main(args):
    from model import builder

    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    print(len(dataset_test))
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                   sampler=test_sampler, num_workers=args.workers,
                                                   collate_fn=utils.collate_func)
    print(args.model)
    single_model = builder.__dict__[args.model](pretrained='',args=args)
    utils.load_model(single_model, args.resume)
    model = single_model.to(device)

    if not args.output_dir:
        args.output_dir = os.path.dirname(args.resume) or '.'
    os.makedirs(args.output_dir, exist_ok=True)
    batch_evaluate(model, data_loader_test, args)

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    args = prepare_official_pretrained_assets(args)
    print('Image size: {}'.format(str(args.img_size)))
    if args.eval_ori_size:
        print('Eval mode: original')
    else:
        print('Eval mode: resized')
    main(args)
