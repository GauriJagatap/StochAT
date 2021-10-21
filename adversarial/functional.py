from typing import Union, Callable, Tuple
from functools import reduce
from collections import deque
from torch.nn import Module
import torch
import numpy as np
from utils import project, random_perturbation, infnorm


def _langevin_samples(model: Module,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        loss_fn: Callable,
                        step: float,
                        eps: float,
                        norm: Union[str, float],
                        ep: float = 1e-3,
                        y_target: torch.Tensor = None,
                        xp: torch.Tensor = None,
                        random: bool = False,
                        clamp: Tuple[float, float] = (0, 1),
                        debug: bool = False,
                        projector: bool = True,
                        gamma: float = 1e-4
                        ) -> torch.Tensor:
    breakflag = False
        
    if random:
        x_adv = x.clone().detach().requires_grad_(True).to(x.device) #initialize x' as x
        x_adv = random_perturbation(x_adv, norm, eps) # initialize x' by adding bounded noise to x
    else:
        x_adv = xp.clone().detach() # initialize by loading previous           

    sb = np.sqrt(x_adv.shape[0])

    #Single Langevin updates 
    _x_adv = x_adv.clone().detach().requires_grad_(True) # _x_adv is current iterate of x' (X'^k)
    prediction = model(_x_adv) # f(x')
    loss = loss_fn(prediction, y) # l(f(x'))
    loss.backward() # grad_x' ( l(f(x')) )
    with torch.no_grad():
        if norm == 2:
            #print(torch.norm(_x_adv.grad))    
            gradx = _x_adv.grad 
            if torch.norm(gradx) < 5e-4:
                breakflag = True
                print('vanishing gradients: change lr')
                
            gradients = _x_adv.grad * 1 / _x_adv.grad.view(_x_adv.shape[0], -1).norm(2, dim=-1)\
                    .view(-1, 1, 1, 1)                
            
            x_adv += step*gradients #x_adv is the new update (X'^(k+1))

            if debug:
                print('2 norm after loss update:',torch.norm(x-_x_adv)/sb)
                
            x_adv += gamma*(x-_x_adv) 
            
            if debug:
                print('2 norm after projection update:',torch.norm(x-x_adv)/sb)
            
            noise = ep *np.sqrt(2*step)*torch.randn_like(_x_adv)
            if debug:
                print('noise:',torch.norm(noise.view(-1))/np.sqrt(noise.detach().cpu().view(-1).size()[0]))
            x_adv += noise
                
            if debug:
                print('2 norm after noise update:',torch.norm(x-x_adv)/sb)
                print('2 norm after corrected noise update:',torch.norm(x+noise-x_adv)/sb)
        elif norm == 'inf':
            gradients = _x_adv.grad.sign() * step
                             
            x_adv += gradients
            if debug:
                print('inf norm of grad step:',infnorm(x-x_adv))
               
            delx = ep * np.sqrt(2 * step) * torch.randn_like(_x_adv)
            x_adv += delx
            if debug:
                print('inf norm of noise:',infnorm(delx))

    if debug:
        x_pre = x_adv.clone().detach().requires_grad_(False).to(x.device)
        if norm == 'inf':
            print('inf norm before project:',infnorm(x-x_pre),'eps=',eps)
        elif norm == 2:
            print('2 norm before project:',torch.norm(x-x_pre)/sb,'eps=',eps)            
            
    if projector:
        x_adv = project(x, x_adv, norm, eps).clamp(*clamp)
        if debug:
            if norm == 'inf':
                print('inf norm after project:',infnorm(x-x_adv),'eps=',eps)
            elif norm == 2:
                print('2 norm after project:',torch.norm(x-x_adv)/sb,'eps=',eps)               
    else:
        x_adv = x_adv.clamp(*clamp)
        if debug:
            if norm == 'inf':
                print('inf norm after clamp:',infnorm(x-x_adv),'eps=',eps)
            elif norm == 2:
                print('2 norm after clamp:',torch.norm(x-x_adv)/sb,'eps=',eps)          

    return x_adv.detach(), breakflag

def entropySmoothing(model: Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        step: float,
        eps: float,
        norm: Union[str, float],
        y_target: torch.Tensor = None,
        xp: torch.Tensor = None,
        random: bool = False,
        gamma: float = 1e-3,
        ep: float = 1e-3,
        clamp: Tuple[float, float] = (0, 1),
        debug: bool = False,           
        projector: bool = False            
                    ) -> torch.Tensor:
        
    """Creates an adversarial sample using SGLD
        x_adv: Adversarially perturbed version of x in kth iteration
    """
    return _langevin_samples(model=model, x=x, y=y, loss_fn=loss_fn, eps=eps, norm=norm, step=step,
                               y_target=y_target, xp=xp, random=random, ep=ep, clamp=clamp, debug=debug, projector=projector, gamma=gamma)
