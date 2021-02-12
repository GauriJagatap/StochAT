from typing import Union, Tuple
from torch.nn import Module
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def project(x: torch.Tensor, x_adv: torch.Tensor, norm: Union[str, int], eps: float) -> torch.Tensor:
    """Projects x_adv into the l_norm ball around x

    Assumes x and x_adv are 4D Tensors representing batches of images

    Args:
        x: Batch of natural images
        x_adv: Batch of adversarial images
        norm: Norm of ball around x
        eps: Radius of ball

    Returns:
        x_adv: Adversarial examples projected to be at most eps
            distance from x under a certain norm
    """
    if x.shape != x_adv.shape:
        raise ValueError('Input Tensors must have the same shape')

    if norm == 'inf':
        # Workaround as PyTorch doesn't have elementwise clip
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    else:
        delta = x_adv - x

        # Assume x and x_adv are batched tensors where the first dimension is
        # a batch dimension
        mask = delta.view(delta.shape[0], -1).norm(norm, dim=1) <= eps

        scaling_factor = delta.view(delta.shape[0], -1).norm(norm, dim=1)
        scaling_factor[mask] = eps

        # .view() assumes batched images as a 4D Tensor
        delta *= eps / scaling_factor.view(-1, 1, 1, 1)

        x_adv = x + delta

    return x_adv


def random_perturbation(x: torch.Tensor, norm: Union[str, int], eps: float) -> torch.Tensor:
    """Applies a random l_norm bounded perturbation to x

    Assumes x is a 4D Tensor representing a batch of images

    Args:
        x: Batch of images
        norm: Norm to measure size of perturbation
        eps: Size of perturbation

    Returns:
        x_perturbed: Randomly perturbed version of x
    """
    perturbation = torch.normal(torch.zeros_like(x), torch.ones_like(x))
    if norm == 'inf':
        perturbation = torch.sign(perturbation) * eps
    else:
        perturbation = project(torch.zeros_like(x), perturbation, norm, eps)

    return x + perturbation


def generate_misclassified_sample(model: Module,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """Generates an arbitrary misclassified sample

    Args:
        model: Model that must misclassify
        x: Batch of image data
        y: Corresponding labels
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

    Returns:
        x_misclassified: A sample for the model that is not classified correctly
    """
    while True:
        x_misclassified = torch.empty_like(x).uniform_(*clamp)

        if model(x_misclassified).argmax(dim=1) != y:
            return x_misclassified

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def infnorm(x):
    infn = torch.max(torch.abs(x.detach().cpu()))
    return infn

def train_adversarial(method,model, device, train_loader, optimizer, epoch,adversary,L,step,eps,norm):
    totalcorrect = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        ypred = model(data)
        
        sgd_loss = nn.CrossEntropyLoss()
        # calculate robust loss per batch
        loss, correct = method(model,optimizer,sgd_loss,data,target,epoch,adversary,L,step,eps,norm)
        totalcorrect += correct
    print('robust train accuracy:',100*totalcorrect/len(train_loader.dataset))   