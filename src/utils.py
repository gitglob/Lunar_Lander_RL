import numpy as np
import torch


def corr(x, y):
    """Compute the correlation between two tensors."""
    cor = torch.corrcoef(torch.stack((x, y)))[0, 1].item()

    return cor

def calculate_step_statistics(old_params, new_params):
    abs_max = 0
    mean_square_value = 0
    count = 0

    for old_param, new_param in zip(old_params, new_params):
        param_change = new_param - old_param
        abs_max = max(abs_max, param_change.abs().max().item())
        mean_square_value += (param_change ** 2).sum().item()
        count += param_change.numel()
    
    mean_square_value /= count

    return abs_max, mean_square_value
    