import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch.utils.data import DataLoader
import yaml
from PIL import Image
import torch.backends.cudnn as cudnn  # 添加这一行

from dataset.semi import PredNoMaskDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default="/home/mmyu/mmyu/wm/data/new_data/roadseg_pred.yaml")
parser.add_argument('--save_path', type=str, default="/home/mmyu/mmyu/wm/data/new_data/pred")
parser.add_argument('--check_point', default="exp/roadseg_semi/unimatch/resnet101-wc/202/best.pth", type=str)
parser.add_argument('--mode', default="pred", type=str)

from tqdm import tqdm  # 导入tqdm库

def evaluate(model, loader, cfg, args):
    model.eval()
    num_files = 0

    with torch.no_grad():
        for img, id in tqdm(loader):  # 在循环外使用tqdm包装loader
            img = img.cuda()
            pred = model(img).argmax(dim=1)

            # 转换预测结果为PIL图像
            for j in range(len(img)):
                pred_image = Image.fromarray(pred[j].cpu().numpy().astype('uint8') * 255)

                # 确保文件夹存在，如果不存在则创建它们
                file_path = f"{args.save_path}/{id[j].split(' ')[0]}"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                pred_image.save(file_path)
                num_files += 1

    return num_files


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    model = DeepLabV3Plus(cfg)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = model.cuda()

    valset = PredNoMaskDataset(cfg['dataset'], cfg['data_root'], cfg['txt_path'], args.mode)

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    if os.path.exists(args.check_point):
        checkpoint = torch.load(args.check_point)
        # 删除模型字典中的 "module" 前缀
        state_dict = checkpoint['model']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 加载新的状态字典
        model.load_state_dict(new_state_dict)

        epoch = checkpoint['epoch']

        num_files = evaluate(model, valloader, cfg, args)

        
        logger.info('***** Prediction***** >>>> : num_files{:.2f}\n'.format(num_files))

if __name__ == '__main__':
    main()
