import argparse
import os
import itertools
import matplotlib.pyplot as plt
import json

import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from Models import modelpool
from Preprocess import datapool
from utils import train, val, seed_all, get_logger, calibrate_thresholds
from Models.layer import *
from Models.VGG import vgg16
from Models.DualVGG import DualVGG

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j','--workers',default=16, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size', default=128, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed', default=44, type=int, help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

parser.add_argument('-data', '--dataset',default='cifar100',type=str,help='dataset: cifa10, cifar100, imagenet')
parser.add_argument('-arch', '--model',default='vgg16',type=str,help='model: resnet34')
parser.add_argument('-id', '--identifier', type=str,help='model statedict identifier')

parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-T', '--time', default=0, type=int, help='snn simulation time')
parser.add_argument('-L', '--L', default=8, type=int, help='Step L')

parser.add_argument('--scaling_factor', default=1.0, type=float, help='scaling factor for the warmup at timestep 1')
parser.add_argument('--mode', default='normal', type=str, help='mode for the IF neuron')
parser.add_argument('--calibrate', action='store_true', help='whether to calibrate thresholds before testing')
parser.add_argument('--calib_epochs', default=5, type=int, help='number of epochs for threshold calibration')
parser.add_argument('--calib_lr', default=0.01, type=float, help='learning rate for threshold calibration')
parser.add_argument('--calib_batch_size', default=None, type=int, help='batch size for calibration (default: same as test batch size)')
parser.add_argument('--calib_samples', default=None, type=int, help='number of samples to use for calibration (default: use all data)')
parser.add_argument('--calib_data', default=None, type=str, help='dataset name for calibration (e.g., cifar100, cifar10). If not specified, use the same dataset as testing')
args = parser.parse_args()

def main():
    global args
    print(args)

    log_dir = f"{args.dataset}-test-accuracy"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    identifier = args.model
    if identifier == 'dualvgg16':
        identifier = 'vgg16'
    identifier += '_L[%d]'%(args.L)

    save_acc_filename = f"{identifier}_L{args.L}_T{args.time}"
    logger = get_logger(os.path.join(log_dir, '%s.log'%(save_acc_filename)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    train_loader, test_loader = datapool(args.dataset, args.batch_size, dist_sample=False)

    model_dir = '%s-checkpoints'% (args.dataset)
    
    # 如果是dualvgg16，需要先加载两个vgg16模型
    if args.model == 'dualvgg16':
        # 确定类别数
        if 'imagenet' in args.dataset.lower():
            num_classes = 1000
        elif '100' in args.dataset.lower():
            num_classes = 100
        else:
            num_classes = 10
        
        # 创建两个vgg16模型
        main_vgg = vgg16(num_classes=num_classes, dropout=0.5)
        aux_vgg = vgg16(num_classes=num_classes, dropout=0.5)
        
        # 加载权重文件
        state_dict = torch.load(os.path.join(model_dir, identifier + '.pth'), map_location=torch.device('cpu'))
        
        # 处理键名转换
        keys = list(state_dict.keys())
        for k in keys:
            if "relu.up" in k:
                state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
            elif "up" in k:
                state_dict[k[:-2]+'thresh'] = state_dict.pop(k)
        
        # 加载权重到两个vgg16模型
        main_vgg.load_state_dict(state_dict)
        aux_vgg.load_state_dict(state_dict)
        
        # 创建dualvgg16模型，传入两个已加载权重的vgg16
        model = DualVGG(vgg_name='VGG16', num_classes=num_classes, dropout=0.5, 
                       fusion_method='none', main_vgg=main_vgg, aux_vgg=aux_vgg)
    else:
        # 对于其他模型，使用原有逻辑
        model = modelpool(args.model, args.dataset)
        
        state_dict = torch.load(os.path.join(model_dir, identifier + '.pth'), map_location=torch.device('cpu'))
        
        keys = list(state_dict.keys())
        for k in keys:
            if "relu.up" in k:
                state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
            elif "up" in k:
                state_dict[k[:-2]+'thresh'] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)

    model.to(device)

    model.set_T(args.time)
    model.set_L(args.L)
    model.set_scaling_factor(args.scaling_factor)
    model.set_mode(args.mode)

    # 阈值校准（如果启用）
    if args.calibrate:
        if args.time == 0:
            logger.warning("校准需要T > 0，但当前T=0，跳过校准")
        else:
            # 确定校准使用的数据集
            calib_dataset_name = args.calib_data if args.calib_data is not None else args.dataset
            logger.info(f"开始阈值校准（使用{calib_dataset_name}数据集）...")
            
            # 准备校准数据加载器
            calib_batch_size = args.calib_batch_size if args.calib_batch_size is not None else args.batch_size
            
            # 使用datapool创建校准数据集的数据加载器
            calib_train_loader, _ = datapool(calib_dataset_name, calib_batch_size, dist_sample=False)
            calib_dataset = calib_train_loader.dataset
            logger.info(f"使用{calib_dataset_name}训练集进行校准，数据集大小: {len(calib_dataset)}")
            
            if args.calib_samples is not None:
                # 如果指定了样本数量，创建一个子集
                from torch.utils.data import Subset
                dataset_size = len(calib_dataset)
                num_samples = min(args.calib_samples, dataset_size)
                calib_indices = torch.randperm(dataset_size)[:num_samples]
                calib_dataset = Subset(calib_dataset, calib_indices)
                logger.info(f"使用子集进行校准，样本数量: {num_samples}")
            
            calib_loader = torch.utils.data.DataLoader(
                calib_dataset,
                batch_size=calib_batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True
            )
            
            # 执行校准
            model = calibrate_thresholds(
                model=model,
                calib_loader=calib_loader,
                device=device,
                epochs=args.calib_epochs,
                lr=args.calib_lr,
                verbose=True
            )
            logger.info("阈值校准完成！")

    acc = val(model, test_loader, args.time, device)

    print(f"Test acc = {acc:.3f}")
    return acc

if __name__ == "__main__":
    main()
