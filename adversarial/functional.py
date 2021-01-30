from typing import Union, Callable, Tuple
from functools import reduce
from collections import deque
from torch.nn import Module
import torch
import numpy as np

from utils import project, generate_misclassified_sample, random_perturbation

def infnorm(x):
    infn = torch.max(torch.abs(x))
    return infn

def shrinkage(x,tau):
    xs = torch.abs(x)-tau
    xs[xs<0] = 0
    return xs

def indicator(x):
    ind = torch.zeros_like(x)
    maxval,maxind = torch.topk(x.view(-1),3000)
    print(maxval[-1])
    indflat = ind.view(-1)
    indflat[maxind]= 1
    ind = torch.reshape(ind,x.shape)
    return ind 

def fgsm(model: Module,
         x: torch.Tensor,
         y: torch.Tensor,
         loss_fn: Callable,
         eps: float,
         clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """Creates an adversarial sample using the Fast Gradient-Sign Method (FGSM)
    """
    x.requires_grad = True
    model.train()
    prediction = model(x)
    loss = -loss_fn(prediction, y)
    loss.backward()

    x_adv = (x - torch.sign(x.grad) * eps).clamp(*clamp).detach()

    return x_adv

# for PGD
def _iterative_gradient(model: Module,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        loss_fn: Callable,
                        k: int,
                        step: float,
                        eps: float,
                        norm: Union[str, float],
                        step_norm: Union[str, float],
                        y_target: torch.Tensor = None,
                        random: bool = False,
                        clamp: Tuple[float, float] = (0, 1),
                        debug: bool = False
                       ) -> torch.Tensor:
    """Base function for PGD and iterated FGSM
    Returns:
        x_adv: Adversarially perturbed version of x
    """
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    targeted = y_target is not None
    sb = np.sqrt(x.shape[0])
    if random:
        x_adv = random_perturbation(x_adv, norm, eps)
        if debug:
            print(torch.norm(x-x_adv)/sb)
    for i in range(k):
        _x_adv = x_adv.clone().detach().requires_grad_(True)

        prediction = model(_x_adv)
        loss = loss_fn(prediction, y_target if targeted else y)
        loss.backward()

        with torch.no_grad():
            if step_norm == 'inf':
                gradients = _x_adv.grad.sign() * step
                                    
            else:
                gradients = _x_adv.grad * step / _x_adv.grad.view(_x_adv.shape[0], -1).norm(step_norm, dim=-1)\
                    .view(-1, 1, 1, 1)

            if targeted:
                x_adv -= gradients
            else:
                # Untargeted: Gradient ascent
                x_adv += gradients
                 
        if debug:
            if step_norm == 2:
                print('2 norm of update:',torch.norm(x-x_adv)/sb,'eps=',eps)
            elif step_norm == 'inf':    
                print('inf norm of update:',infnorm(x-x_adv),'eps=',eps)
                
        # Project back into l_norm ball and correct range
        x_adv = project(x, x_adv, norm, eps).clamp(*clamp)
        if debug:
            if step_norm == 2:
                print('2 norm of update:',torch.norm(x-x_adv)/sb,'eps=',eps)
            elif step_norm == 'inf':    
                print('inf norm of update:',infnorm(x-x_adv),'eps=',eps)
             
    return x_adv.detach()


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
    x_adv = x.clone().detach().requires_grad_(True).to(x.device) #load x_adv (samples from p(x')) as x itself and modify it to form init 
    targeted = y_target is not None
    sb = np.sqrt(x_adv.shape[0])
    
    if random:
        x_adv = random_perturbation(x_adv, norm, eps) # initialize x' by adding bounded noise to x
    else:
        x_adv = xp.clone().detach() # initialize by loading previous           
        
    #Single Langevin updates 
    _x_adv = x_adv.clone().detach().requires_grad_(True) # _x_adv is current iterate of x' 
    prediction = model(_x_adv) # f(x')
    loss = loss_fn(prediction, y_target if targeted else y) # l(f(x'))
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
            x_adv += step*gradients

            if debug:
                print('2 norm after loss update:',torch.norm(x-x_adv)/sb)
                
            x_adv += gamma*(x-_x_adv)             
            if debug:
                print('2 norm after projection update:',torch.norm(x-x_adv)/sb)
            
            noise = ep *np.sqrt(2*step)*torch.randn_like(_x_adv)
            x_adv += noise
                
            if debug:
                print('2 norm after noise update:',torch.norm(x-x_adv)/sb)
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
    #else:
    #    x_adv = x_adv.clamp(*clamp)
    #    if debug:
    #        if norm == 'inf':
    #            print('inf norm after clamp:',infnorm(x-x_adv),'eps=',eps)
    #        elif norm == 2:
    #            print('2 norm after clamp:',torch.norm(x-x_adv)/sb,'eps=',eps)          

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
        gamma: float = 1e-4,
        ep: float = 1e-6,
        clamp: Tuple[float, float] = (0, 1),
        debug: bool = False,           
        projector: bool = False            
                    ) -> torch.Tensor:
        
    """Creates an adversarial sample using the Projected Gradient Descent Method

    This is a white-box attack.

    Args:
        model: Model
        x: Batch of samples
        y: Corresponding labels
        loss_fn: Loss function to maximise
        k: Number of iterations to make
        step: Size of step (i.e. L2 norm) to make at each iteration
        eps: Maximum size of adversarial perturbation, larger perturbations will be projected back into the
            L_norm ball
        norm: Type of norm
        random: Whether to start PGD within a random point in the l_norm ball
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

    Returns:
        x_adv: Adversarially perturbed version of x
    """
    return _langevin_samples(model=model, x=x, y=y, loss_fn=loss_fn, eps=eps, norm=norm, step=step,
                               y_target=y_target, xp=xp, random=random, ep=ep, clamp=clamp, debug=debug, projector=projector, gamma=gamma)

def iterated_fgsm(model: Module,
                  x: torch.Tensor,
                  y: torch.Tensor,
                  loss_fn: Callable,
                  k: int,
                  step: float,
                  eps: float,
                  norm: Union[str, float],
                  y_target: torch.Tensor = None,
                  random: bool = False,
                  clamp: Tuple[float, float] = (0, 1),
                  debug: bool = False
                 ) -> torch.Tensor:
    """Creates an adversarial sample using the iterated Fast Gradient-Sign Method

    This is a white-box attack.

    Args:
        model: Model
        x: Batch of samples
        y: Corresponding labels
        loss_fn: Loss function to maximise
        k: Number of iterations to make
        step: Size of step to make at each iteration
        eps: Maximum size of adversarial perturbation, larger perturbations will be projected back into the
            L_norm ball
        norm: Type of norm
        y_target:
        random: Whether to start Iterated FGSM within a random point in the l_norm ball
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

    Returns:
        x_adv: Adversarially perturbed version of x
    """
    return _iterative_gradient(model=model, x=x, y=y, loss_fn=loss_fn, k=k, eps=eps, norm=norm, step=step,
                               step_norm='inf', y_target=y_target, random=random, clamp=clamp, debug=debug)


def pgd(model: Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        k: int,
        step: float,
        eps: float,
        norm: Union[str, float],
        y_target: torch.Tensor = None,
        random: bool = False,
        clamp: Tuple[float, float] = (0, 1),
        debug: bool =False
       ) -> torch.Tensor:
    """Creates an adversarial sample using the Projected Gradient Descent Method

    This is a white-box attack.

    Args:
        model: Model
        x: Batch of samples
        y: Corresponding labels
        loss_fn: Loss function to maximise
        k: Number of iterations to make
        step: Size of step (i.e. L2 norm) to make at each iteration
        eps: Maximum size of adversarial perturbation, larger perturbations will be projected back into the
            L_norm ball
        norm: Type of norm
        random: Whether to start PGD within a random point in the l_norm ball
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

    Returns:
        x_adv: Adversarially perturbed version of x
    """
    return _iterative_gradient(model=model, x=x, y=y, loss_fn=loss_fn, k=k, eps=eps, norm=norm, step=step, step_norm='inf',
                               y_target=y_target, random=random, clamp=clamp, debug=debug)


def boundary(model: Module,
             x: torch.Tensor,
             y: torch.Tensor,
             k: int,
             orthogonal_step: float = 1e-2,
             perpendicular_step: float = 1e-2,
             initial: torch.Tensor = None,
             clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """Implements the boundary attack

    This is a black box attack that doesn't require knowledge of the model
    structure. It only requires knowledge of

    https://arxiv.org/pdf/1712.04248.pdf

    Args:
        model: Model to be attacked
        x: Batched image data
        y: Corresponding labels
        k: Number of steps to take
        orthogonal_step: orthogonal step size (delta in paper)
        perpendicular_step: perpendicular step size (epsilon in paper)
        initial: Initial attack image to start with. If this is None then use random noise
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

    Returns:
        x_adv: Best i.e. closest adversarial example for x
    """
    orth_step_stats = deque(maxlen=30)
    perp_step_stats = deque(maxlen=30)
    # Factors to adjust step sizes by
    orth_step_factor = 0.97
    perp_step_factor = 0.97

    def _propose(x: torch.Tensor,
                 x_adv: torch.Tensor,
                 y: torch.Tensor,
                 model: Module,
                 clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
        """Generate proposal perturbed sample

        Args:
            x: Original sample
            x_adv: Adversarial sample
            y: Label of original sample
            clamp: Domain (i.e. max/min) of samples
        """
        # Sample from unit Normal distribution with same shape as input
        perturbation = torch.normal(torch.zeros_like(x_adv), torch.ones_like(x_adv))

        # Rescale perturbation so l2 norm is delta
        perturbation = project(torch.zeros_like(perturbation), perturbation, norm=2, eps=orthogonal_step)

        # Apply perturbation and project onto sphere around original sample such that the distance
        # between the perturbed adversarial sample and the original sample is the same as the
        # distance between the unperturbed adversarial sample and the original sample
        # i.e. d(x_adv, x) = d(x_adv + perturbation, x)
        perturbed = x_adv + perturbation
        perturbed = project(x, perturbed, 2, torch.norm(x_adv - x, 2)).clamp(*clamp)

        # Record success/failure of orthogonal step
        orth_step_stats.append(model(perturbed).argmax(dim=1) != y)

        # Make step towards original sample
        step_towards_original = project(torch.zeros_like(perturbation), x - perturbed, norm=2, eps=perpendicular_step)
        perturbed = (perturbed + step_towards_original).clamp(*clamp)

        # Record success/failure of perpendicular step
        perp_step_stats.append(model(perturbed).argmax(dim=1) != y)

        # Clamp to domain of sample
        perturbed = perturbed.clamp(*clamp)

        return perturbed

    if x.size(0) != 1:
        # TODO: Attack a whole batch in parallel
        raise NotImplementedError

    if initial is not None:
        x_adv = initial
    else:
        # Generate initial adversarial sample from uniform distribution
        x_adv = generate_misclassified_sample(model, x, y)

    total_stats = torch.zeros(k)

    for i in range(k):
        # Propose perturbation
        perturbed = _propose(x, x_adv, y, model, clamp)

        # Check if perturbed input is adversarial i.e. gives the wrong prediction
        perturbed_prediction = model(perturbed).argmax(dim=1)
        total_stats[i] = perturbed_prediction != y
        if perturbed_prediction != y:
            x_adv = perturbed

        # Check statistics and adjust step sizes
        if len(perp_step_stats) == perp_step_stats.maxlen:
            if torch.Tensor(perp_step_stats).mean() > 0.5:
                perpendicular_step /= perp_step_factor
                orthogonal_step /= orth_step_factor
            elif torch.Tensor(perp_step_stats).mean() < 0.2:
                perpendicular_step *= perp_step_factor
                orthogonal_step *= orth_step_factor

        if len(orth_step_stats) == orth_step_stats.maxlen:
            if torch.Tensor(orth_step_stats).mean() > 0.5:
                orthogonal_step /= orth_step_factor
            elif torch.Tensor(orth_step_stats).mean() < 0.2:
                orthogonal_step *= orth_step_factor

    return x_adv


def _perturb(x: torch.Tensor,
             p: float,
             i: int,
             j: int,
             clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """Perturbs a pixel in an image

    Args:
        x: image
        p: perturbation parameters
        i: row
        j: column
    """
    if x.size(0) != 1:
        raise NotImplementedError('Only implemented for single image')

    x[0, :, i, j] = p * torch.sign(x[0, :, i, j])

    return x.clamp(*clamp)


def local_search(model: Module,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 k: int,
                 branching: Union[int, float] = 0.1,
                 p: float = 1.,
                 d: int = None,
                 clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """Performs the local search attack

    This is a black-box (score based) attack first described in
    https://arxiv.org/pdf/1612.06299.pdf

    Args:
        model: Model to attack
        x: Batched image data
        y: Corresponding labels
        k: Number of rounds of local search to perform
        branching: Either fraction of image pixels to search at each round or
            number of image pixels to search at each round
        p: Size of perturbation
        d: Neighbourhood square half side length

    Returns:
        x_adv: Adversarial version of x
    """
    if x.size(0) != 1:
        # TODO: Attack a whole batch at a time
        raise NotImplementedError('Only implemented for single image')

    x_adv = x.clone().detach().requires_grad_(False).to(x.device)
    model.eval()

    data_shape = x_adv.shape[2:]
    if isinstance(branching, float):
        branching = int(reduce(lambda x, y: x*y, data_shape) * branching)

    for _ in range(k):
        # Select pixel locations at random
        perturb_pixels = torch.randperm(reduce(lambda x, y: x*y, data_shape))[:branching]

        perturb_pixels = torch.stack([perturb_pixels // data_shape[0], perturb_pixels % data_shape[1]]).transpose(1, 0)

        # Kinda hacky but works for MNIST (i.e. 1 channel images)
        # TODO: multi channel images
        neighbourhood = x_adv.repeat((branching, 1, 1, 1))
        perturb_pixels = torch.cat([torch.arange(branching).unsqueeze(-1), perturb_pixels], dim=1)
        neighbourhood[perturb_pixels[:, 0], 0, perturb_pixels[:, 1], perturb_pixels[:, 2]] = 1

        predictions = model(neighbourhood).softmax(dim=1)
        scores = predictions[:, y]

        # Select best perturbation and continue
        i_best, j_best = perturb_pixels[scores.argmin(dim=0).item(), 1:]
        x_adv[0, :, i_best, j_best] = 1.
        x_adv.clamp_(*clamp)

        # Early exit if adversarial is found
        worst_prediction = predictions.argmax(dim=1)[scores.argmin(dim=0).item()]
        if worst_prediction.item() != y.item():
            return x_adv

    # Attack failed, return sample with lowest score of correct class
    return x_adv


'''
def _langevin_shrinkage(model: Module,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        loss_fn: Callable,
                        k: int,
                        step: float,
                        eps: float,
                        norm: Union[str, float],
                        step2: float,
                        gamma: float,
                        ep: float,
                        y_target: torch.Tensor = None,
                        xp: torch.Tensor = None,
                        random: bool = False,
                        clamp: Tuple[float, float] = (0, 1),
                        debug: bool = False,
                        projector: bool = True
                        ) -> torch.Tensor:
    
    x_adv = x.clone().detach().requires_grad_(True).to(x.device) # load x_adv (samples from p(x')) as x itself and modify it to form init 
    targeted = y_target is not None
    sb = np.sqrt(x_adv.shape[0])
    if random:
        x_adv = random_perturbation(x_adv, norm, eps) # initialize x' by adding bounded noise to x
    else:
        x_adv = xp.clone().detach()
        if debug:
            print(infnorm(x-xp))
        
    for i in range(k): # Langevin updates 
        _x_adv = x_adv.clone().detach().requires_grad_(True) # _x_adv is current iterate of x' 

        prediction = model(_x_adv) # f(x')
        loss = loss_fn(prediction, y_target if targeted else y) # l(f(x'))
        loss.backward() # grad_x' ( l(f(x')) )
        with torch.no_grad():
            if norm == 2:
                gradients = _x_adv.grad * 1 / _x_adv.grad.view(_x_adv.shape[0], -1).norm(2, dim=-1)\
                    .view(-1, 1, 1, 1)                
                x_adv += step*gradients
                
                if debug:
                    print('gradient:',torch.norm(gradients))
                    print('2 norm after loss update:',torch.norm(x-x_adv)/sb)
                
                x_adv += gamma*(x-_x_adv)             
                if debug:
                    print('2 norm after projection update:',torch.norm(x-x_adv)/sb)
            
                noise = ep *np.sqrt(2*step)*torch.randn_like(_x_adv)
                x_adv += noise
                
                if debug:
                    print('2 norm after noise update:',torch.norm(x-x_adv)/sb)
            else:
                #shrink = False ## ignore cw type update for now
                #if shrink:
                #    x_delinf = shrinkage(x-_x_adv,gamma)
                #    _x_adv += step2 * x_delinf
                gradients1 = _x_adv.grad.sign() * step
                             
                x_adv += gradients1
                if debug:
                    print('inf norm of grad step:',infnorm(x-x_adv))
               
                delx = ep * np.sqrt(2 * step) * torch.randn_like(_x_adv)
                x_adv += delx
                if debug:
                    print('inf norm of noise:',infnorm(delx))

    if debug:
        x_pre = x_adv.clone().detach().requires_grad_(False).to(x.device)
        print('inf norm before project:',infnorm(x-x_pre),'eps=',eps)
    if projector:
        x_adv = project(x, x_adv, norm, eps).clamp(*clamp)
        if debug:
            print('inf norm after project:',infnorm(x-x_adv),'eps=',eps)
    else:
        x_adv = x_adv.clamp(*clamp)

    return x_adv.detach()
'''