"""Training-only proposal allocation with an unchanged ray/primitive budget."""
import torch


def select_seed_rays(population, requested, seed, scores=None):
    if not (isinstance(population, int) and isinstance(requested, int)
            and population >= 0 and requested > 0):
        raise ValueError('Valid ray population and budget required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    count = min(population, requested)
    if scores is not None:
        scores = scores.detach().to(device='cpu', dtype=torch.float64)
        if scores.shape != (population,) or not torch.isfinite(scores).all() or (scores < 0).any():
            raise ValueError('Aligned finite nonnegative training RGB scores required')
    if scores is None or count == population or not bool(scores.sum() > 0):
        return torch.randperm(population, generator=generator)[:count]
    # A quarter of sampling probability remains uniform. Scores propose where
    # to search; they do not confer leaf existence, depth truth or opacity.
    scaled = scores/scores.max()
    probability = .25/population+.75*scaled/scaled.sum()
    return torch.multinomial(probability, count, replacement=False, generator=generator)
