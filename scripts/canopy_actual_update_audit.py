"""Measure actual optimizer/postprocessing movement, not gradient intent alone."""
import torch


class ActualUpdateAudit:
    def __init__(self, named_parameters):
        self.parameters = dict(named_parameters)
        self.before = {k:p.detach().cpu().clone() for k,p in self.parameters.items()}
        self.gradients = {k:p.grad.detach().cpu().clone() for k,p in self.parameters.items() if p.grad is not None}
        if any(not torch.isfinite(g).all() for g in self.gradients.values()):
            raise ValueError('Finite declared-objective gradients required')

    @torch.no_grad()
    def measure(self):
        result = {}; total_dot = 0.
        for name,p in self.parameters.items():
            current=p.detach().cpu();delta=current-self.before[name]; grad=self.gradients.get(name)
            if not torch.isfinite(delta).all(): raise ValueError('Nonfinite actual parameter update')
            flat=delta.reshape(len(delta),-1)
            changed=flat.ne(0).any(1)
            record=dict(changed_rows=int(changed.sum()),max_abs_delta=float(delta.abs().max()))
            if grad is not None:
                dot=float((grad.double()*delta.double()).sum()); total_dot+=dot
                inactive=~grad.reshape(len(grad),-1).ne(0).any(1)
                record.update(gradient_dot_delta=dot,zero_gradient_changed_rows=int((inactive&changed).sum()))
            if name.endswith('logits'):
                # Stable tau=softplus(logit), including near-opaque states.
                dt=torch.nn.functional.softplus(current)-torch.nn.functional.softplus(self.before[name])
                record.update(tau_increased_rows=int((dt>0).sum()),tau_decreased_rows=int((dt<0).sum()),
                              tau_delta_sum=float(dt.double().sum()))
            result[name]=record
        return dict(gradient_dot_delta=total_dot,parameters=result,
                    warning='Local first-order direction only; rerender the same objective to measure actual descent')
