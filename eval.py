import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch.utils.data import DataLoader
import yaml
import torch.backends.cudnn as cudnn  # 添加这一行

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default="/home/mmyu/mmyu/wm/UniMatch/configs/roadseg_semi.yaml")
parser.add_argument('--save-path', type=str, default="/home/mmyu/mmyu/wm/UniMatch/exp/eval")
parser.add_argument('--check_point', default="/home/mmyu/mmyu/wm/UniMatch/exp/roadseg_semi/unimatch/resnet101-wc/202/best.pth", type=str)


def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    # 用于计算每一类
    class_recall = np.zeros(cfg['nclass'])  # 用于记录每一类的召回率
    class_precision = np.zeros(cfg['nclass'])  # 用于记录每一类的精度

    # 用于累积整个测试集的TP、FP和FN
    total_tp = np.zeros(cfg['nclass'])
    total_fp = np.zeros(cfg['nclass'])
    total_fn = np.zeros(cfg['nclass'])

    with torch.no_grad():
        for img, mask, id in loader:
            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img).argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            
            for i in range(cfg['nclass']):
                # 计算每一类的混淆矩阵
                tp = np.sum((mask.cpu().numpy() == i) & (pred.cpu().numpy() == i))
                fp = np.sum((mask.cpu().numpy() != i) & (pred.cpu().numpy() == i))
                fn = np.sum((mask.cpu().numpy() == i) & (pred.cpu().numpy() != i))

                # 累积整个测试集的TP、FP和FN
                total_tp[i] += tp
                total_fp[i] += fp
                total_fn[i] += fn
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    # 计算整个测试集的精度和召回率
    print(total_tp)
    print(total_fp)
    class_recall = total_tp / (total_tp + total_fn + 1e-10) * 100
    class_precision = total_tp / (total_tp + total_fp + 1e-10) * 100
    return mIOU, iou_class, class_recall, class_precision

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    model = DeepLabV3Plus(cfg)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = model.cuda()

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    if os.path.exists(args.check_point):
        checkpoint = torch.load(args.check_point)
        # 删除模型字典中的 "module" 前缀
        state_dict = checkpoint['model']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 加载新的状态字典
        model.load_state_dict(new_state_dict)

        epoch = checkpoint['epoch']

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class, class_accuracy, class_recall = evaluate(model, valloader, eval_mode, cfg)

        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            
        for (cls_idx, accuracy) in enumerate(class_accuracy):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'Accuracy: {:.6f}%'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], accuracy))
            
        for (cls_idx, recall) in enumerate(class_recall):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'Recall: {:.6f}%'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], recall))
            
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))

if __name__ == '__main__':
    main()
