"""
Cifar-10 WideResNet-34
"""

import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets
from multiprocessing import cpu_count
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import olympic
from typing import Union, Callable, Tuple
import sys
sys.path.append('../adversarial/')
sys.path.append('../architectures/')
from functional import boundary, iterated_fgsm, local_search, pgd, entropySmoothing
from ESGD_utils import *
import pickle
import time
import torch.backends.cudnn as cudnn
import argparse, math, random
import ESGD_optim
from pathlib2 import Path
from wideresnet import WideResNet
# Net = WideResNet
# NetName = 'WideResNet'


## Reproducibility
seed = 1
torch.set_num_threads(2)
if DEVICE=='cuda':
    torch.cuda.set_device(-1)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', '-o', help='output directory', default='./models')
    parser.add_argument('--batchsize', '-bs', help='Batch size', default=128, type=int)
    parser.add_argument('--save_freq', '-sf', help='Save epochs', default=5, type=int)
    parser.add_argument('--lr', '-lr', help='Learning rate', default=0.1, type=float)
    parser.add_argument('--wd', '-wd', help='weight decay', default=0.95, type-float)
    parser.add_argument('--dset', '-ds', help='Dataset', choices=['cifar10'], default='cifar10')
    parser.add_argument('--epochs', '-e', help='Epochs', default=10000, type=int)
    parser.add_argument('--cuda', choices=['single', 'multi'], defualt='single')
    return parser

def evaluate(model, device, loader, mode='train'):
    model.eval()
    loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(loader.dataset)
    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        mode, loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))
    accuracy = correct / len(loader.dataset)
    return loss, accuracy

def adjust_learning_rate(optimizer, epoch,lr_init):
    """decrease the learning rate"""
    lr = lr_init
    if epoch >= 75:
        lr = lr_init * 0.1
    if epoch >= 90:
        lr = lr_init * 0.01
    if epoch >= 100:
        lr = lr_init * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args):

    device  = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    dataset = 'CIFAR10' # [MNIST, CIFAR10]
    transform = transforms.Compose([transforms.ToTensor(),])
    bsz = args.batchsize
    train = datasets.CIFAR10('../../data/CIFAR10', train=True, transform=transform, download=True)
    val = datasets.CIFAR10('../../data/CIFAR10', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train, batch_size=bsz, shuffle=True, **kwargs)
    val_loader = DataLoader(val, batch_size=bsz, shuffle=False, **kwargs)
    net = WideResNet(depth=34).to(device)

    if args.cuda == 'multi':
        net = nn.DataParallel(net)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    scheduler = StepLR(optimizer, step_size=40, gamma=0.1) 

    for epoch in range(1, epochs + 1):
        print('Epoch:',epoch)
        
        for img, label in train_loader:
            optimizer.zero_grad()
            
            img = img.to(device)
            label = label.to(device)
            pred = net(img)

            loss_val = F.cross_entropy(pred, label)
            loss_val.backward()
            optimizer.step()
            scheduler.step()

        if epoch % 5 == 0:
            trainloss, train_acc = evaluate(model, device, train_loader, mode='train')
            valloss, val_acc = evaluate(model, device, val_loader, model='val')

        if epoch % args['save_freq'] == 0:
            torch.save(model_SATInf.state_dict(),
                os.path.join(model_dir, 'model-nn-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                os.path.join(model_dir, 'opt-nn-checkpoint_epoch{}.tar'.format(epoch)))

def main():
    args = build_parser().parse_args()
    train(args)


