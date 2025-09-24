import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.cpp_extension import load
import gc
import os
import time
import math
from functools import partial
from typing import Dict, List, Optional, Tuple, Callable, Union
from torch._higher_order_ops.foreach_map import foreach_map
    
@torch.compile(mode='reduce-overhead')
def nd_zeropower_via_newtonschulz6(G: torch.Tensor, dtype) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ∈ [1 - l, 1 + r], which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    
    X = G.bfloat16()
    
    if G.ndim == 2:
        return F.rms_norm(X, (X.size(-1),), eps=1e-8).to(dtype)
        
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    for a, b, c in [        
        (3.8623, -8.1113, 4.8906),
        (3.6474, -6.5244, 3.3818),
        (3.7099, -6.3466, 3.1357),
        (3.9248, -6.2353, 2.8378),
        (2.6142, -2.9580, 1.1347),
        (2.1210, -1.7900, 0.6660),
    ]:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = F.rms_norm(X, (X.size(-2), X.size(-1),), eps=1e-8)
    
    return X.to(dtype)
        
class AdaMuon(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: Union[float, Callable[[torch.Tensor], torch.Tensor]] = 1e-3,
        betas: tuple[float, float] = (0.8, 0.95),
        weight_decay: float = 1e-2,
        nesterov: bool = True,
        backend: str = "newtonschulz",
    ):
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, 
                        nesterov=nesterov, backend=backend)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            m_bufs = []
            v_bufs = []

            lr = group['lr']
            decay = group['weight_decay']
            
            for p in group['params']:
                dtype = p.dtype
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad.bfloat16())
                    state = self.state[p]
                    if 'm_buf' not in state:
                        state['m_buf'] = torch.zeros_like(p, dtype=torch.bfloat16, memory_format=torch.preserve_format)
                    if 'v_buf' not in state:
                        state['v_buf'] = torch.zeros_like(p, dtype=torch.bfloat16, memory_format=torch.preserve_format)
                    m_bufs.append(state['m_buf'])
                    v_bufs.append(state['v_buf'])

            if not params_with_grad:
                continue

            beta1, beta2 = group['betas']
            
            grad_sqs = torch._foreach_mul(grads, grads)
            
            torch._foreach_lerp_(m_bufs, grads, 1 - beta1)
            torch._foreach_lerp_(v_bufs, grad_sqs, 1 - beta2)
            
            v_sqrt = torch._foreach_sqrt(v_bufs)
            torch._foreach_add_(v_sqrt, 1e-18)

            grads = torch._foreach_div(m_bufs, v_sqrt)
            
            grads = list(nd_zeropower_via_newtonschulz6(torch.stack(grads, dim=0), dtype))

            torch._foreach_mul_(params_with_grad, 1 - lr*decay)
            torch._foreach_add_(params_with_grad, grads, alpha=-lr)

        return loss

def CosineAnnealingWarmup(optimizer, init_lr, base_lr, anneal_lr, warmup_steps, anneal_steps):
    a = optim.lr_scheduler.LinearLR(optimizer, 
                                    start_factor = init_lr/base_lr, 
                                    end_factor   = 1.0, 
                                    total_iters  = warmup_steps)
    
    b = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                             T_max      = anneal_steps, 
                                             eta_min    = anneal_lr, 
                                             last_epoch = 0)
    
    c = optim.lr_scheduler.ConstantLR(optimizer, 
                                      factor      = anneal_lr/base_lr, 
                                      total_iters = 100_000_000)
    
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, 
                                                schedulers=[a, b, c], 
                                                milestones=[warmup_steps, 
                                                            warmup_steps+anneal_steps])
    return scheduler