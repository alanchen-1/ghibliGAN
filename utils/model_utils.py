from collections import OrderedDict
import torch.nn as nn
import os

def print_network(net : nn.Module, verbose : bool = True):
    if verbose:
        print(net)

    num_params = 0
    for parameter in net.parameters():
        num_params += parameter.numel()
    
    num_params /= 1e6
    print(f'Total params:  {num_params:.3f} million')

def print_losses(losses : OrderedDict, epoch : int, total_epochs : int, iters : int, total_iters : int, subset : list = None):
    if subset is None:
        subset = losses.keys()
    
    to_print = f"Losses [{epoch} / {total_epochs}] [{iters} / {total_iters}]:"
    for key in subset:
        to_print += f" [{key} : {losses[key]:.4f}] "
    
    print(to_print)

def get_latest_num(checkpoints_dir : str):
    files = list(filter(lambda name: os.path.splitext(name)[1] == '.pth', os.listdir(checkpoints_dir)))
    mx = -1
    for filename in files:
        prefix = filename.split('_')[0]
        if (prefix != 'latest'):
            try:
                if int(prefix) > mx:
                    mx = int(prefix)
            except ValueError:
                pass
    return mx


