import os
import torch


def get_save_subdir(folder_dir):
    # Check if the folder_dir exists, if not, create it
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
        
    # List all items in the given directory
    items = os.listdir(folder_dir)
    
    # Filter out only subfolders
    subfolders = [item for item in items if os.path.isdir(os.path.join(folder_dir, item))]
    
    # Convert subfolder names to integers if possible
    subfolder_ints = []
    for subfolder in subfolders:
        try:
            subfolder_ints.append(int(subfolder))
        except ValueError:
            # If conversion fails, just ignore the subfolder
            continue
    
    # Sort the list of integers
    subfolder_ints.sort()

    # Find the first missing integer
    current = 0
    for num in subfolder_ints:
        if num != current:
            return str(current)
        current += 1
    
    return os.path.join(folder_dir, str(current))

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
    