"""
Attacks on WRN-34
"""

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import argparse
import os
import sys
import numpy as np
sys.path.append('../adversarial/')
sys.path.append('../architectures/')
from wideresnet import WideResNet
from foolbox.models import PyTorchModel
from foolbox.attacks import LinfDeepFoolAttack, LinfProjectedGradientDescentAttack, L2ProjectedGradientDescentAttack, L2CarliniWagnerAttack, L2DeepFoolAttack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LinfCarliniWagnerAttack(L2CarliniWagnerAttack):
    """
    Implements Carlini-Wagner Linf Attack
    """
    def __init__(self):
        super().__init_()
    

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Model Path', required=True)
    parser.add_argument('-at', '--att_type', help='Attack type', choices=['cwlinf', 'cwl2', 'deepfooll2', 'deepfoollinf', 'pgdlinf', 'pgdl2'], default='deepfool')
    parser.add_argument('-o', '--outfile', help='OUtput file', default='./results.txt')
    parser.add_argument('-eps', '--epsilon', help='max epsilon ', default=(8/255.0), type=float)
    parser.add_argument('-ss', '--stepsize', help='Step size', default=0.0007, type=float)
    return parser

def get_dataloader(batchsize=64):
    custom_transform = transforms.Compose([transforms.ToTensor()])
    ds = CIFAR10('../../data/cifar10', train=False, transform=custom_transform, download=True)
    return DataLoader(ds, batchsize=batchsize, shuffle=False)

def get_attack(args):
    if args.att_type == 'cwl2':
        return L2CarliniWagnerAttack(binary_search_steps=10, steps=10000, stepsize = args.stepsize, confidence=0, abort_early=True)
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

    model = WideResNet(depth=34, num_classes=10)
    model.load_state_dict(torch.load(args.model))

    model = model.to(device)
    tgt_model = PyTorchModel(model)
    attack = get_attack(args)
    dl = get_dataloader(batchsize=args.batchsize)
    if 'linf' in args.att_type:
        order = np.inf
    elif 'l2' in args.att_type:
        order = 2

    successes = 0.0
    epsilons = np.zeros(args.batchsize)
    for idx, (imgs, labels) in enumerate(dl):
        imgs = imgs.to(device)
        labels = labels.to(device)
        _, adv_examples, succ = attack(tgt_model, imgs, labels, epsilons=args.epsilon)
        successes += succ
        epses = np.linalg.norm(imgs - adv_examples, ord=order, dim=0)
        epsilons[idx:idx+args.batchsize] = epses
    rob_accuracy = 1 - successes/10000.0
    with open(args.outfile, 'w') as f:
        f.write(rob_accuracy+'\n')
    np.save('epsilons'+args.outfile, epsilons)





