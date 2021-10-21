#!/usr/bin/env python
# coding: utf-8

# Main script for training a classifier for CIFAR-10 using l_inf ATENT [Table 3 and Figure 3 of paper].
# 
# Notebook contains printed result from evaluation of pretrained model

# # Adversarial Training via ENTropic regularization (l_inf ATENT)

# ### SoTA - collected from various papers

# vanila SGD: 
# MNIST - 99%+ (most cnns), CIFAR10 - 93%+ (resnet18), 96%+ (wideresnet) 
# 
# MNIST:
# 
# adversarial attacks: 
# l-inf @ eps = 80/255 PGD @20 steps: TRADES - 96.07%, MART 96.48%, MMA 95.25%, PGD - 96.01% (7 layer cnn)
# 
# adversarial attacks:
# l-2 @ eps = 2 PGD @40 steps: MMA - 73.02
# 
# CIFAR10:
# 
# adversarial attacks: 
# l-inf @ eps = 8/255 PGD @20 steps: 
# (WideResNet) TRADES 53-56% - (WRN-34-10), MART 57-58% (WRN-34-10), MMA 47%, PGD 48% - (WRN-32-10)// 49% - (WRN-34-10), Std - 0.03%
# 
# Benign accuracies: 
# (WideResNet)TRADES 84.92%, MART 83.62%, MMA 84.36, PGD 87.14%, Std 95.8% 
# 
# adversarial attacks: 
# l-inf @ eps = 8/255 @20 steps: 
# (ResNet10) TRADES 45.4%, MART 46.6%, MMA 37.26%, PGD 42.27%, Std 0.14%
# 
# adversarial attacks:
# l-2 @ eps = 128/255: MMA 67%, PGD 68%

# Reference repos for baselines:
# TRADES :
# https://github.com/yaodongyu/TRADES (MNIST: small cnn, CIFAR10: WideResNet34)
# MMA : 
# https://github.com/BorealisAI/mma_training (MNIST: lenet5, CIFAR10: WideResNet28)
# MART :
#  https://github.com/YisenWang/MART (CIFAR10: ResNet18 and WideResNet34)
# PGD: (CIFAR10: ResNet50) https://github.com/MadryLab/robustness 

# ### IMPORT LIBRARIES

# In[1]:


import sys,os
sys.path.append('../adversarial/')
sys.path.append('../architectures/')
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from torchvision import transforms, datasets

import numpy as np
import matplotlib.pyplot as plt


# ### IMPORT UTILITIES

# In[2]:


from functional import entropySmoothing
from utils import eval_train, eval_test, infnorm, train_adversarial


# ### SET TRAINING PARAMETERS

# In[3]:


args = {}
#data loading
args['seed'] = 1
args['test_batch_size'] = 128
args['train_batch_size'] = 128
kwargs = {'num_workers': 4, 'pin_memory': True}
args['no_cuda'] = False

if not args['no_cuda']:
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
else:
    DEVICE = 'cpu'

# params for SGLD (inner loop)
args['attack'] = 'l_inf'
args['norm'] = 'inf'
args['epsilon'] = 0.031
args['num_steps'] = 10
args['step_size'] = 0.007
args['random'] =True

# params for SGD (outer loop)
args['lr'] = 0.1
args['momentum'] = 0.9
args['weight_decay'] = 5e-4
args['epochs'] = 76
args['save_freq'] = 1

# load model
args['pretrained'] = False 


# ### LOAD DATA

# In[4]:


dataset = 'CIFAR10' # [MNIST, CIFAR10]

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
    
train_loader = DataLoader(train, batch_size=args['test_batch_size'], shuffle=True, **kwargs)
val_loader = DataLoader(val, batch_size=args['train_batch_size'], shuffle=False, **kwargs)


# ### LOAD NETWORK

# In[5]:


if dataset=='CIFAR10':
    #[ResNet18,ResNet34,ResNet50,WideResNet]
    from resnet import ResNet18,ResNet34,ResNet50
    from wideresnet import WideResNet
    Net = WideResNet
    NetName = 'WideResNet34'


# ### SET RANDOM SEED 

# In[6]:


torch.set_num_threads(2)
if DEVICE=='cuda':
    torch.cuda.set_device(-1)
    torch.cuda.manual_seed(args['seed'])
    cudnn.benchmark = True
random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])


# ### WHITEBOX L-INF ATTACK

# In[7]:


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args['epsilon'],
                  num_steps=20,
                  step_size=0.003
                 ):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args['random']:
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
    with torch.no_grad():
        loss_pgd = nn.CrossEntropyLoss()(model(X_pgd), y)
    return err, err_pgd, loss_pgd.item()

def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    lossrob = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust, losspgd = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
        lossrob = lossrob + losspgd
    rob = 100-100*robust_err_total.item()/len(test_loader.dataset)   
    lossrob /= len(test_loader)
    print('robust test loss:',lossrob)
    print('natural_acc_total: ', 100-100*natural_err_total.item()/len(test_loader.dataset))
    print('robust_acc_total: ', rob)    
    return rob


# ### L-INF ATENT MODULE

# In[8]:


def adversarial_training_entropy(model, optimiser, loss_fn, x, y, epoch, adversary, L, step, eps, norm):
    model.train()
    
    # Adversial perturbation
    alpha=0.9
    loss = 0
    
    for l in range(L):     
        
        if l==0: # initialize using random perturbation of true x, run for one epoch
            random=True
            xp = None
            projector=False
        elif l>0 and l<L-1: # initialize with previous iterate of adversarial perturbation, run one epoch
            random=False
            xp=x_adv
            projector = False
        elif l == L-1: # initialize with previous iterate, run one epoch, project to epsilon ball
            random=False
            xp = x_adv
            projector=True
            
        x_adv,bfl = adversary(model, x, y, loss_fn, xp=xp, step=step, eps=eps, norm=norm, random=random, ep=1e-3,projector=projector)
        
        optimiser.zero_grad()
        y_pred = model(x_adv)
        pred = y_pred.max(1, keepdim=True)[1]
        correct = pred.eq(y.view_as(pred)).sum().item()
        loss = (1-alpha)*loss + alpha*loss_fn(y_pred, y)
        
    loss.backward()
    optimiser.step()
    return loss, correct


# ### INITIALIZE NET OR LOAD FROM CHECKPOINT

# In[9]:


model_ATENT = Net().to(DEVICE)
model_ATENT = nn.DataParallel(model_ATENT)
#load pretrained state dict here
if args['pretrained']:
    pathstr = '../trainedmodels/CIFAR10/BEST_model-nn-epoch76-robacc57.pt 
    model_ATENT.load_state_dict(torch.load(pathstr))
    rob = eval_adv_test_whitebox(model_ATENT, DEVICE, val_loader)            


# ### ADJUST LEARNING RATE SCHEDULER

# In[10]:


def adjust_learning_rate(optimizer, epoch,lr_init):
    """decrease the learning rate"""
    lr = lr_init
    if epoch >= 75:
        lr = lr_init * 0.1 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ### PATHS TO STORE MODELS

# In[11]:


if not os.path.exists('../trainedmodels'): #path to save final trained model
    os.makedirs('../trainedmodels')
model_dir = '../trainedmodels/'+NetName #path to save each epoch while training
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# ### TRAIN MODULE USING ATENT

# In[12]:


if not args['pretrained']:    
    ## training params
    lr_init = args['lr']
    optimizer = optim.SGD(model_ATENT.parameters(), lr=lr_init, momentum=args['momentum'], 
                          weight_decay=args['weight_decay'] )

    ## train model
    for epoch in range(1, args['epochs']+1):
        print('Epoch:',epoch)

        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch,lr_init)

        # adversarial training
        train_adversarial(adversarial_training_entropy,model_ATENT, DEVICE, train_loader, optimizer, 
                          epoch,adversary=entropySmoothing,L=args['num_steps'],step=args['step_size'],
                          eps=args['epsilon'],norm=args['norm'])

        # evaluation on natural and adversarial examples
        print('================================================================')
        eval_train(model_ATENT, DEVICE, train_loader)
        rob = eval_adv_test_whitebox(model_ATENT, DEVICE, val_loader)            
        print('================================================================')

        # save checkpoint
        if (epoch-1) % args['save_freq'] == 0:
            torch.save(model_ATENT.state_dict(),
                       os.path.join(model_dir, 'model-nn-epoch{}-robacc{}.pt'.format(epoch,int(np.round(rob)))))

    # save final model
    modelname = '../trainedmodels/'+dataset+'/'+NetName+'_ATENTInf_ep'+str(epochs)+'_lr'+str(lr_init)+'_robacc_'+str(int(np.round(rob)))+'.pt'
    torch.save(model_ATENT,modelname)

