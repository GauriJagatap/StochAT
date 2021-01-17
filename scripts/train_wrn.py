"""
Cifar-10 WideResNet-34
"""

import numpy as np
import pandas as pd
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
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
from wideresnet import WideResNet
# Net = WideResNet
# NetName = 'WideResNet'


## Reproducibility


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', '-o', help='output directory', default='./models')
    parser.add_argument('--batchsize', '-bs', help='Batch size', default=128, type=int)
    parser.add_argument('--save_freq', '-sf', help='Save epochs', default=5, type=int)
    parser.add_argument('--lr', '-lr', help='Learning rate', default=0.1, type=float)
    parser.add_argument('--wd', '-wd', help='weight decay', default=0.0005, type=float)
    parser.add_argument('--dset', '-ds', help='Dataset', choices=['cifar10'], default='cifar10')
    parser.add_argument('--epochs', '-e', help='Epochs', default=1000, type=int)
    parser.add_argument('--cuda', choices=['single', 'multi'], default='single')
    return parser

def evaluate(model, device, loader, mode='train'):
    model.eval()
    loss = 0.0
    accuracy = 0.0
    correct = 0.0
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

    model_dir = args.outdir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    torch.set_num_threads(2)
    if device=='cuda':
        torch.cuda.set_device(-1)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    kwargs = {'num_workers': 4, 'pin_memory': True}

    epochs = args.epochs
    dataset = 'CIFAR10' # [MNIST, CIFAR10]
    transform = transforms.Compose([transforms.ToTensor(),])
    bsz = args.batchsize
    train = datasets.CIFAR10('../../data/CIFAR10', train=True, transform=transform, download=True)
    val = datasets.CIFAR10('../../data/CIFAR10', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train, batch_size=bsz, shuffle=True, **kwargs)
    val_loader = DataLoader(val, batch_size=bsz, shuffle=False, **kwargs)
    net = WideResNet(depth=34).to(device)
    loss = nn.CrossEntropyLoss()
    if args.cuda == 'multi':
        net = nn.DataParallel(net)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    scheduler = StepLR(optimizer, step_size=15, gamma=0.1) 

    for epoch in range(epochs):
        print('Epoch:',epoch)
        net.train()
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            pred = net(img)

            loss_val = loss(pred, label)
            loss_val.backward()
            optimizer.step()
        
        scheduler.step()

        if epoch % 1 == 0:
            trainloss, train_acc = evaluate(net, device, train_loader, mode='train')
            valloss, val_acc = evaluate(net, device, val_loader, mode='val')
            print(f'Epoch:{epoch}, train loss:{trainloss}, train_acc:{train_acc}, val loss:{valloss}, val acc:{val_acc}')

        if epoch % args.save_freq == 0:
            torch.save(net.state_dict(),
                os.path.join(model_dir, 'model-nn-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                os.path.join(model_dir, 'opt-nn-checkpoint_epoch{}.tar'.format(epoch)))

def main():
    args = build_parser().parse_args()
    train(args)

if __name__ == '__main__':
    main()
