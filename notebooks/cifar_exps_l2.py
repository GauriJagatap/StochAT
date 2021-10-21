#!/usr/bin/env python
# coding: utf-8

# Main script for training a classifier for CIFAR-10 using l_2 ATENT [Table 4 and Table 5 of paper]
# 
# Notebook contains printed result from evaluation of pretrained model

# # Adversarial Training via ENTropic regularization (l_2 ATENT)

# ### SoTA

# vanila SGD: 
# MNIST - 99%+ (most cnns), CIFAR10 - 93%+ (resnet18), 96%+ (wideresnet) 
# 
# MNIST:
# 
# adversarial attacks:
# l-2 @ eps = 2 PGD @40 steps: MMA - 73.02
# 
# CIFAR10:
# 
# adversarial attacks:
# l-2 @ eps = 128/255: MMA 67%, PGD 68%

# Reference repos for baselines: TRADES : https://github.com/yaodongyu/TRADES (MNIST: small cnn, CIFAR10: WideResNet34) MMA : https://github.com/BorealisAI/mma_training (MNIST: lenet5, CIFAR10: WideResNet28) MART : https://github.com/YisenWang/MART (CIFAR10: ResNet18 and WideResNet34) PGD: (CIFAR10: ResNet50) https://github.com/MadryLab/robustness

# ### IMPORT LIBRARIES

# In[1]:


import sys,os
sys.path.append('../adversarial/')
sys.path.append('../architectures/')
import random
from time import perf_counter

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
from utils import eval_train, eval_test, infnorm, train_adversarial, project
from cifar_resnet import resnet as resnet_cifar


# ### SET TRAINING PARAMETERS

# In[3]:


args = {}
#data loading
args['seed'] = 2
args['test_batch_size'] = 256
args['train_batch_size'] = 256
kwargs = {'num_workers': 4, 'pin_memory': True}
args['no_cuda'] = False

if not args['no_cuda']:
    if torch.cuda.is_available():
        DEVICE = 'cuda:1'
    else:
        DEVICE = 'cpu'
else:
    DEVICE = 'cpu'

# params for SGLD (inner loop)
args['attack'] = 'l_2'
args['norm'] = 2
args['epsilon'] = 0.435
args['num_steps'] = 20 #vary this
args['step_size'] = 0.5 * args['epsilon']/args['num_steps']
args['random'] = True

# params for SGD (outer loop)
args['lr'] = 0.1
args['momentum'] = 0.9
args['weight_decay'] = 1e-4
args['epochs'] = 54
args['save_freq'] = 1

# load model
args['pretrained'] = False


# ### LOAD DATA

# In[4]:


dataset = 'CIFAR10' # [MNIST, CIFAR10]

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

train_loader = DataLoader(train, batch_size=args['train_batch_size'], shuffle=True, **kwargs)
val_loader = DataLoader(val, batch_size=args['test_batch_size'], shuffle=False, **kwargs)


# ### INITIALIZE NETWORK

# In[5]:


if dataset=='CIFAR10':
    #[ResNet18,ResNet34,ResNet50,WideResNet]
    from resnet import ResNet18,ResNet34,ResNet50
    from wideresnet import WideResNet
    Net = ResNet18
    #Net = resnet_cifar(depth=20, num_classes=10)
    NetName = 'ResNet18'


# ### RANDOM SEED 

# In[6]:


seed = args['seed']
torch.set_num_threads(2)
if DEVICE=='cuda':
    torch.cuda.set_device(-1)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# ### WHITEBOX L-2 ATTACK

# In[7]:


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args['epsilon'],
                  norm=args['norm'],
                  num_steps=args['num_steps'],
                  step_size=args['step_size']):
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
        if norm=='inf':
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
        elif norm==2:
            eta = step_size * X_pgd.grad.data / X_pgd.grad.view(X_pgd.shape[0], -1).norm(2, dim=-1)                    .view(-1, 1, 1, 1)
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            X_pgd = project(X, X_pgd, norm, epsilon)            
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    
    with torch.no_grad():
        loss_pgd = nn.CrossEntropyLoss()(model(X_pgd), y)
    #print('err pgd (white-box): ', err_pgd)
    return err, err_pgd, loss_pgd.item()

def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    lossrob  = 0
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


# ### L-2 ATENT MODULE

# In[8]:


def adversarial_training_entropy(model, optimiser, loss_fn, x, y, epoch, adversary, L, step, eps, norm):
    """Performs a single update against a specified adversary"""
    model.train()
    
    # Adversial perturbation
    alpha=0.9
    loss = 0
    gamma = 0.05
    projector = False #(always false for l2 setup)
    
    for l in range(L):     
        
        if l==0: ## initialize using random perturbation of true x, run for one epoch
            k=1
            random=True
            xp = None
        elif l>0 and l<L-1: ## initialize with previous iterate of adversarial perturbation, run one epoch
            k=1
            random=False
            xp=x_adv
        elif l == L-1: ## initialize with previous iterate, run one epoch, project to epsilon ball
            k=1
            random=False
            xp = x_adv
        #eps=0.577 produces effective noise std 0.12    
        x_adv,bfl = adversary(model, x, y, loss_fn, xp=xp, step=step, eps=eps, norm=norm, random=random, ep=0.577,projector=projector,gamma=gamma, debug=False)
        optimiser.zero_grad()
        y_pred = model(x_adv)
        pred = y_pred.max(1, keepdim=True)[1]
        correct = pred.eq(y.view_as(pred)).sum().item()
        
        loss_lg = loss_fn(y_pred, y)
        loss = (1-alpha)*loss + alpha*loss_lg
        
        if bfl: #break and readjust learning rate if gradients vanish
            break
            
    loss.backward()
    optimiser.step()
    return loss, correct


# ### INITIALIZE NET OR LOAD FROM CHECKPOINT

# In[9]:


model_ATENT = Net().to(DEVICE)
#model_ATENT = nn.DataParallel(model_ATENT)
#load pretrained state dict here
if args['pretrained']:
    pathstr = '../trainedmodels/CIFAR10/smooth-model-l2.pt' 
    model_ATENT.load_state_dict(torch.load(pathstr))
    rob = eval_adv_test_whitebox(model_ATENT, DEVICE, val_loader)         


# ### ADJUST LEARNING RATE SCHEDULER

# In[10]:


def adjust_learning_rate(optimizer, epoch,lr_init):
    """decrease the learning rate"""
    lr = lr_init
    if epoch >= 50:   
        lr = lr_init * 0.1
    print('lr:',lr)    
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
    epochs = args['epochs']  
    lr_init = args['lr']
    #optimizer = optim.Adam(model_ATENT.parameters(), lr=lr_init)

    optimizer = optim.SGD(model_ATENT.parameters(), lr=lr_init, momentum=0.9, weight_decay=1e-4)
    ## train model

    for epoch in range(epochs+30):
        print('Epoch:',epoch)

        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch,lr_init)

        # adversarial training
        tic = perf_counter()
        train_adversarial(adversarial_training_entropy,model_ATENT, DEVICE, train_loader, 
                          optimizer, epoch,adversary=entropySmoothing,L=args['num_steps'],
                          step=args['step_size'],eps=args['epsilon'],norm=args['norm'])
        toc = perf_counter()
        # evaluation on natural and adversarial examples
        print('Time elapsed:%f', tic, toc, toc-tic)
        print('================================================================')
        # eval_train(model_ATENT, DEVICE, train_loader)
        # rob = eval_adv_test_whitebox(model_ATENT, DEVICE, val_loader)            
        print('================================================================')

        # save checkpoint
        # torch.save(model_ATENT.state_dict(),
        #                os.path.join(model_dir, 'model-nn-epoch{}-robacc{}.pt'.format(epoch,int(np.round(rob)))))

    ## save model

    # modelname = '../trainedmodels/'+dataset+'/'+NetName+'_ATENT_'+args['attack']+'_ep'+str(epochs)+'_lr'+str(lr_init)+'.pt'
    # torch.save(model_ATENT,modelname)

