"""Immutable, resumable full-horizon view schedule, independent of global RNG."""
import torch


def training_schedule(views,steps,*,seed=1701,reshuffle=False):
    if not views or steps<=0 or len(set(views))!=len(views):
        raise ValueError('Unique nonempty views and positive horizon required')
    generator=torch.Generator().manual_seed(seed)
    result=[];first=None
    while len(result)<steps:
        if first is None or reshuffle:
            order=[views[i] for i in torch.randperm(len(views),generator=generator).tolist()]
            if first is None:first=order
        else:order=first
        result.extend(order)
    return result[:steps]
