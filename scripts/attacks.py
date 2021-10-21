"""
Attacks on WRN-34
"""

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
import setGPU
import argparse
import os
import sys
import numpy as np
sys.path.append('../adversarial/')
sys.path.append('../architectures/')
from wideresnet import WideResNet
from small_cnn import SmallCNN
from advertorch_examples.models import LeNet5Madry
from foolbox.models import PyTorchModel
from foolbox.attacks import LinfDeepFoolAttack, LinfProjectedGradientDescentAttack, L2ProjectedGradientDescentAttack, L2CarliniWagnerAttack, L2DeepFoolAttack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
    

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Model Path', required=True)
    parser.add_argument('-at', '--att_type', help='Attack type', choices=['cwlinf', 'cwl2', 'deepfooll2', 'deepfoollinf', 'pgdlinf', 'pgdl2'], default='deepfool')
    parser.add_argument('-o', '--outfile', help='OUtput file', default='./results.txt')
    parser.add_argument('-eps', '--epsilon', help='max epsilon ', default=(8/255.0), type=float)
    parser.add_argument('-ss', '--stepsize', help='Step size', default=0.0007, type=float)
    parser.add_argument('--batchsize', '-bs', help='batchsize', default=128, type=int)
    return parser

def get_dataloader(batchsize=64):
    custom_transform = Compose([ToTensor()])
    ds = MNIST('../../data/mnist_test', train=False, transform=custom_transform, download=True)
    return DataLoader(ds, batch_size=batchsize, shuffle=False)

def get_attack(args):
    if args.att_type == 'cwl2':
        return L2CarliniWagnerAttack(binary_search_steps=10, steps=1000, stepsize = args.stepsize, confidence=0, abort_early=True)
    elif args.att_type == 'cwlinf':
        return NotImplementedError('Not implemented yet!')
    elif args.att_type == 'deepfooll2':
        return L2DeepFoolAttack()
    elif args.att_type == 'deepfoolinf':
        return LinfDeepFoolAttack()
    elif args.att_type == 'pgdlinf':
        return L2ProjectedGradientDescentAttack(rel_stepsize=args.stepsize)
    elif args.att_type == 'pgdl2':
        return LinfProjectedGradientDescentAttack(rel_stepsize=args.stepsize)

def main():
    args = build_parser().parse_args()

    #model = WideResNet(depth=34, num_classes=10)1
    model = LeNet5Madry()
    model.load_state_dict(torch.load(args.model))
    #model = torch.load(args.model)
    model.eval()

    model = model.to(device)
    tgt_model = PyTorchModel(model, bounds=(0,1))
    attack = get_attack(args)
    dl = get_dataloader(batchsize=args.batchsize)
    if 'linf' in args.att_type:
        order = np.inf
    elif 'l2' in args.att_type:
        order = 2

    successes = 0.0
    correct = 0.0
    epsilons = np.zeros(10000)
    for idx, (imgs, labels) in enumerate(dl):
        print(idx)
        imgs = imgs.to(device)
        labels = labels.to(device)
        _, adv_examples, succ = attack(tgt_model, imgs, labels, epsilons=args.epsilon)
        successes += succ.sum()
        #epses = torch.norm(imgs.view(-1, 28*28) - adv_examples.view(-1, 28*28), p=order, dim=1)
        #epsilons[idx*args.batchsize:(idx+1)*args.batchsize] = epses.cpu()
    rob_accuracy = 1 - successes/10000.0
    print('rob_accuracy', rob_accuracy)
    # with open(args.outfile, 'w') as f:
    #     f.write(rob_accuracy+'\n')
    #np.save('epsilons'+args.outfile, epsilons)


if __name__ == '__main__':
    main()