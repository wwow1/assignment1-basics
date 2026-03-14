from typing import Any, Optional, Callable
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0 or betas[0] >= 1:
            raise ValueError(f"Invalid beta[0] value: {betas[0]}")
        if betas[1] < 0 or betas[1] >= 1:
            raise ValueError(f"Invalid beta[1] value: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.grad)
                    state['v'] = torch.zeros_like(p.grad)

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                t = state.get("t", 1)
                beta1 = group['betas'][0]
                beta2 = group['betas'][1]
                # Update biased first moment estimate
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * grad**2
                lr = group['lr']
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                # Update biased second moment estimate
                p.data -= lr_t * state['m'] / (torch.sqrt(state['v']) + group['eps'])
                # Apply weight decay
                p.data -= group['weight_decay'] * lr * p.data
                state['t'] = t + 1
