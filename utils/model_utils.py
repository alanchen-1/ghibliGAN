from tkinter import W
import torch.nn as nn

def print_network(net : nn.Module, verbose : bool = True):
    if verbose:
        print(net)

    num_params = 0
    for parameter in net.parameters():
        num_params += parameter.numel()
    
    num_params /= 1e6
    print(f'Total params:  {num_params:.3f} million')


