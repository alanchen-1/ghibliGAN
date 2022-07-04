# defines useful utilities for other files to use
from collections import OrderedDict
import torch.nn as nn
import os

def print_network(net : nn.Module, verbose : bool = True):
    """
    Prints network if verbose and the number of parameters.
        Parameters:
            net (nn.Module) : network to print
            verbose (bool) : print network architecture or not
    """
    if verbose:
        print(net)

    num_params = 0
    for parameter in net.parameters():
        num_params += parameter.numel()
    
    num_params /= 1e6
    print(f'Total params:  {num_params:.3f} million')

def print_losses(losses : OrderedDict, epoch : int, total_epochs : int, iters : int, total_iters : int, subset : list = None):
    """
    Prints losses at end of every dataloader iteration.
        Parameters: 
            losses (OrderedDict) : ordered dictionary of losses
            epoch (int) : current epoch
            total_epochs (int) : total number of epochs
            iters (int) : number of iterations
            total_iters (int) : total number of iterations
            subset (list) : subset of keys, if desired. default is None
    """
    if subset is None:
        subset = losses.keys()
    
    to_print = f"Losses [{epoch} / {total_epochs}] [{iters} / {total_iters}]:"
    for key in subset:
        to_print += f" [{key} : {losses[key]:.4f}] "
    
    print(to_print)

def get_latest_num(checkpoints_dir : str):
    """
    Calculates the epoch corresponding to the "latest" saved checkpoint.
    Assumes there exists such a checkpoint, its saved as a .pth file, and that the files have been named 
    such the epoch they were saved at is a prefix before the first underscore (ex. "{epoch}_{rest of name}.pth").
        Parameters:
            checkpoints_dir (str) : directory of checkpoints
        Returns:
            mx (int) : epoch corresponding to the "latest" checkpoint
    """
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


