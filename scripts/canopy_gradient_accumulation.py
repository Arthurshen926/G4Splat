"""Bounded microbatch windows measured in rendered training images."""


def accumulation_window(image_step, image_budget, views_per_update):
    if not all(isinstance(x, int) for x in (image_step, image_budget, views_per_update)):
        raise ValueError('Integer accumulation schedule required')
    if not (1 <= image_step <= image_budget and 1 <= views_per_update <= 8):
        raise ValueError('Bounded positive accumulation schedule required')
    start = ((image_step-1)//views_per_update)*views_per_update+1
    end = min(start+views_per_update-1, image_budget)
    return image_step == start, image_step == end, end-start+1
