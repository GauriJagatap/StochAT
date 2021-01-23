#!/usr/bin/env python
# coding: utf-8

# Evaluate the models

# # Stochastic Adversarial Training (StochAT)

# ### SoTA

# vanila SGD: 
# MNIST - 99%+ (most cnns), CIFAR10 - 93%+ (resnet18), 96%+ (wideresnet) 
# 
# MNIST:
# 
# adversarial attacks: 
# l-inf @ eps = 80/255 @20 steps: TRADES - 96.07% - (4 layer cnn), MART 96.4%, MMA 95.5%, PGD - 96.01% - (4 layer cnn)
# 
# adversarial attacks:
# l-2 @ eps = 32/255 (check): TRADES, MMA, PGD
# 
# CIFAR10:
# 
# adversarial attacks: 
# l-inf @ eps = 8/255 @20 steps: 
# TRADES 53-56% - (WRN-34-10), MART 57-58% (WRN-34-10), MMA 47%, PGD 48% - (WRN-32-10)// 49% - (WRN-34-10), Std - 0.03%
# https://openreview.net/pdf?id=rklOg6EFwS (Table 4)
# 
# adversarial attacks: 
# l-inf @ eps = 8/255 @20 steps: 
# [ResNet10] TRADES 45.4%, MART 46.6%, MMA 37.26%, PGD 42.27%, Std 0.14%
# 
# Benign accuracies: TRADES 84.92%, MART 83.62%, MMA 84.36, PGD 87.14%, Std 95.8% [wideresnet]
# https://openreview.net/pdf?id=Ms9zjhVB5R (Table 1)
# 
# adversarial attacks:
# l-2 @ eps = 32/255 (check): TRADES, MART, MMA, PGD
# 
# TBD: CWinf attacks

# ## Pretrained models for comparison

# download pretrained models and place in ../trainedmodels/MNIST or ../trainedmodels/CIFAR10 respectively
# 
# ### TRADES :
# https://github.com/yaodongyu/TRADES (MNIST: small cnn, CIFAR10: WideResNet34)
# ### MMA : 
# https://github.com/BorealisAI/mma_training (MNIST: lenet5, CIFAR10: WideResNet28)
# ### MART :
#  https://github.com/YisenWang/MART (CIFAR10: ResNet18 and WideResNet34)

# ## IMPORT LIBRARIES
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
from trades import trades_loss
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=0.031,
                  num_steps=20,
                  step_size=0.003,
                  random=True
                  ):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(DEVICE)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    #print('err pgd (white-box): ', err_pgd)
    return err, err_pgd

def eval_adv_test_whitebox(model, device, test_loader, args):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, epsilon=args.epsilon, step_size=args.step_size, random=args.random)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_acc_total: ', 100-natural_err_total.item()/100)
    print('robust_acc_total: ', 100-robust_err_total.item()/100)


def infnorm(x):
    infn = torch.max(torch.abs(x.detach().cpu()))
    return infn


def train_adversarial(method,model, device, train_loader, optimizer, epoch,adversary,k,step,eps,norm,random):
    totalcorrect = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        ypred = model(data)
        
        sgd_loss = nn.CrossEntropyLoss()
        # calculate robust loss per batch
        loss, correct = method(model,optimizer,sgd_loss,data,target,epoch,adversary,k,step,eps,norm,random)
        totalcorrect += correct
    print('robust train accuracy:',100*totalcorrect/len(train_loader.dataset))   


def adjust_learning_rate(optimizer, epoch,lr_init):
    """decrease the learning rate"""
    lr = lr_init
    if epoch >= 50:
        lr = lr_init * 0.1
    if epoch >= 75:
        lr = lr_init * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# # TRAIN MODEL USING SAT

def adversarial_training_entropy(model, optimiser, loss_fn, x, y, epoch, adversary, k, step, eps, norm, random):
    """Performs a single update against a specified adversary"""
    model.train()
    
    # Adversial perturbation
    N = k
    alpha=0.95
    loss = 0
    advs = []
    for l in range(N):
        
        if l==0:
            k=1
            random=True
            xp = None
            projector=False
        elif l>0 and l<N-1:
            k=1
            random=False
            xp=x_adv
            projector = False
        elif l == N-1:
            k=1
            random=False
            xp = x_adv
            projector=True
            
        x_adv = adversary(model, x, y, loss_fn, xp=xp, k=k, step=step, step2=0.05, eps=eps, norm=norm, random=random, gamma=0.3, ep=1e-3,projector=projector, debug=False)
        optimiser.zero_grad()
        y_pred = model(x_adv)
        pred = y_pred.max(1, keepdim=True)[1]
        correct = pred.eq(y.view_as(pred)).sum().item()
        loss = (1-alpha)*loss + alpha*loss_fn(y_pred, y)
        
    loss.backward()
    optimiser.step()
    return loss, correct

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
    parser.add_argument('--modelpath', '-mpath', help='Path to model', required=True)
    parser.add_argument('--epsilon', '-eps', default=0.031, type=float, help='Epsilon value for defense')
    parser.add_argument('--step_size', '-ss', default=0.007, type=float, help='Step size')
    parser.add_argument('--random', action='store_true', help='random start')
    #parser.add_argument
    return parser

def main():

    args = build_parser().parse_args()

    kwargs = {'num_workers': 4, 'pin_memory': True}

    # args = {}
    # args['test_batch_size'] = 128
    # args['train_batch_size'] = 128
    # args['no_cuda'] = False
    # args['epsilon'] = 0.031
    # args['num_steps'] = 10
    # args['step_size'] = 0.007
    # args['random'] =True,
    # args['white_box_attack']=True
    # args['log_interval'] = 100
    # args['beta'] = 6.0
    # args['seed'] = 1
    # args['lr'] = 0.1
    # args['momentum'] = 0.9
    # args['epochs'] = 5
    # args['batch_size'] = 128
    # args['save_freq'] = 3

    dataset = 'CIFAR10' # [MNIST, CIFAR10]
    if dataset == 'MNIST':
        transform = transforms.Compose([
        transforms.ToTensor()])
        train = datasets.MNIST('../../data/MNIST', train=True, transform=transform, download=True)
        val = datasets.MNIST('../../data/MNIST', train=False, transform=transform, download=True)
    elif dataset == 'CIFAR10':
        # setup data loader
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        train = datasets.CIFAR10('../../data/CIFAR10', train=True, transform=transform_train, download=True)
        val = datasets.CIFAR10('../../data/CIFAR10', train=False, transform=transform_test, download=True)
        
    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, **kwargs)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, **kwargs)

    if dataset=='MNIST':
        from small_cnn import SmallCNN   
        Net = SmallCNN
        NetName = 'SmallCNN'

    if dataset=='CIFAR10':
        #[ResNet18,ResNet34,ResNet50,WideResNet]
        from resnet import ResNet18,ResNet34,ResNet50
        from wideresnet import WideResNet
        Net = WideResNet
        NetName = 'WideResNet'


    seed = 1
    torch.set_num_threads(2)
    if DEVICE=='cuda':
        torch.cuda.set_device(-1)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    #model_dir = '../WRN_ATENT'

    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)

    model_SATInf = Net().to(DEVICE)
    #w = 155
    mpath = args.modelpath
    model_SATInf.load_state_dict(torch.load(mpath))
    eval_train(model_SATInf, DEVICE, train_loader)

    ## initialize model
    #model_SATInf = Net().to(DEVICE)
    #model_SATInf = nn.DataParallel(model_SATInf)
    ## training params
    epochs = args.epochs
    lr_init = args.lr
    optimizer = optim.SGD(model_SATInf.parameters(), lr=lr_init, momentum=0.9, weight_decay=2e-4)
    ## train model

    for epoch in range(epochs):
        print('Epoch:',epoch)
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch,lr_init)
        print('LR:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        # adversarial training
        train_adversarial(adversarial_training_entropy,model_SATInf, DEVICE, train_loader, optimizer, epoch,adversary=entropySmoothing,k=10,step=0.007,eps=0.031,norm='inf',random=False)
        #train_adversarial(adversarial_training,model_SATInf, DEVICE, train_loader, optimizer, epoch,adversary=iterated_fgsm,k=40,step=0.01,eps=0.3,norm='inf',random=False)

        # evaluation on natural examples

        if (epoch) % args.save_freq == 0:
            print('================================================================')
            eval_train(model_SATInf, DEVICE, train_loader)
            #eval_test(model_SATInf, DEVICE, val_loader)
            eval_adv_test_whitebox(model_SATInf, DEVICE, val_loader, args)            
            print('================================================================')
        #scheduler.step()


        # save checkpoint
        if (epoch) % args.save_freq == 0:
            torch.save(model_SATInf.state_dict(),
                    os.path.join(args.outdir, 'model-nn-epoch{}.pt'.format(epoch)))

    ## save model
    modelname = args.outdir+'/'+NetName+'_SATInf_ep'+str(epoch)+'_lr'+str(lr_init)+'.pt'
    torch.save(model_SATInf,modelname)

if __name__ == '__main__':
    main()