"""Experimental Adam with per-row clocks; exact-zero-gradient rows are paused.

This is not a visibility oracle or a correction to standard Adam mathematics.
It changes the treatment of intermittently observed rows and needs a control.
"""
import torch


class RowActiveAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=.001, betas=(.9, .999), eps=1e-15):
        if not (0 < lr and 0 <= betas[0] < 1 and 0 <= betas[1] < 1 and eps > 0):
            raise ValueError('Positive Adam rate/epsilon and valid moments required')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.ndim < 1 or p.is_complex(): raise ValueError('Real row-structured parameters required')
                if p.grad is not None and (p.grad.is_sparse or not torch.isfinite(p.grad).all()):
                    raise ValueError('Finite dense gradients required before any update')
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                active = grad.reshape(len(p), -1).ne(0).any(1)
                if not active.any(): continue
                state = self.state[p]
                if not state:
                    state['step'] = torch.zeros(len(p), dtype=torch.long, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'][active] += 1
                m = state['exp_avg'][active]*beta1+(1-beta1)*grad[active]
                v = state['exp_avg_sq'][active]*beta2+(1-beta2)*grad[active].square()
                state['exp_avg'][active] = m; state['exp_avg_sq'][active] = v
                count = state['step'][active].to(dtype=torch.float64)
                shape = (-1,)+(1,)*(p.ndim-1)
                bc1 = (1-beta1**count).to(p).reshape(shape)
                bc2 = (1-beta2**count).to(p).reshape(shape)
                denominator = v.sqrt()/bc2.sqrt()+group['eps']
                p[active] = p[active]-group['lr']*m/bc1/denominator
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p)
                if not state: continue
                step = state['step']
                if (step.shape != (len(p),) or not torch.isfinite(step).all()
                        or (step < 0).any() or (step != step.round()).any()
                        or state['exp_avg'].shape != p.shape or state['exp_avg_sq'].shape != p.shape):
                    raise ValueError('Per-row Adam state identity required')
                state['step'] = step.to(device=p.device, dtype=torch.long)
