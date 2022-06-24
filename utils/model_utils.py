from collections import OrderedDict
import torch.nn as nn

def print_network(net : nn.Module, verbose : bool = True):
    if verbose:
        print(net)

    num_params = 0
    for parameter in net.parameters():
        num_params += parameter.numel()
    
    num_params /= 1e6
    print(f'Total params:  {num_params:.3f} million')

def print_losses (losses : OrderedDict, epoch : int, total_epochs : int, iters : int, total_iters : int, subset : list = None):
    if subset is None:
        subset = losses.keys()
    
    to_print = f"Losses [{epoch} / {total_epochs}] [{iters} / {total_iters}]:"
    for key in subset:
        to_print += f" [{key} : {losses[key]:.4f}] "
    
    print(to_print)


