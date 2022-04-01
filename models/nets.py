import torch
import torch.nn as nn
from torch.nn import init
from blocks import *

def init_params(network : nn.Module, type : str, scale : float):
    def init_params_helper(model : nn.Module):
        if type == 'normal':
            init.normal_(network.weight.data, 0, scale)
        elif type == 'xavier':
            init.xavier_normal_(network.weight.data, gain=scale)
        elif type == 'kaiming':
            init.kaiming_normal_(model.weight.data)
        
        # init bias
        if hasattr(model, 'bias'):
            init.constant_(model.bias.data, 0.0)
    # apply to network 
    network.apply(init_params_helper)

def add_device(network : nn.Module, use_gpu : bool = True):
    """
    Attaches model to device to allow for GPU training.
    Currently only attaches to one GPU (since I only have one),
    but easily expandable to incorporate multiple GPU support using Data.Parallel (maybe a future update)
        Parameters:
            network (nn.Module) : network to move to a device
            use_gpu (bool) : whether to use GPU (if available, default is True)
        Returns:
            network (nn.Module) : original network attached to the specified device
    """
    device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")
    network.to(device)
    return network

def init_model(network : nn.Module, use_gpu : bool = True, init_type : str = 'normal', init_scale : float = 0.02):
    network = add_device(network, use_gpu)
    init_params(network, type=init_type, scale=init_scale)
    return network