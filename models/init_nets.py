import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from blocks import ResNetGenerator, PatchDiscriminator
from utils.str_to_layer import create_norm

def init_params(network : nn.Module, type : str, scale : float):
    def init_params_helper(model : nn.Module):
        if type == 'normal':
            init.normal_(network.weight.data, 0, scale)
        elif type == 'xavier':
            init.xavier_normal_(network.weight.data, gain=scale)
        elif type == 'kaiming':
            init.kaiming_normal_(model.weight.data)
        else:
            raise NotImplementedError(f'{type} initialization not implemented')
        
        # init bias to 0
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
    """
    Initializes a model by adding it to a device and initializing the weights.
        Parameters:
            network (nn.Module) : network to initialize
            use_gpu (bool) : whether to use GPU in training
            init_type (str) : initialization type
            init_scale (float) : initialization scale, only applicable for normal or xavier
        Returns:
            (nn.Module) : initialized network
    """
    network = add_device(network, use_gpu)
    init_params(network, type=init_type, scale=init_scale)
    return network

def init_resnet_generator(in_channels : int, out_channels : int, num_blocks : int, use_gpu : bool = True, init_type : str = 'normal', init_scale : float = 0.02):
    """
    Define and initialize a simple ResNetGenerator (uses a lot of default values).
    Provides a template to define custom initalize methods like this.

        Parameters:
            in_channels (int) : channels in inputs
            out_channels (int) : channels in outputs
            num_blocks (int) : number of ResNetBlocks
            use_gpu (bool) : whether to use GPU in training
            init_type (str) : initialization type
            init_scale (float) : initialization scale, only applicable for normal or xavier
        Returns:
            (nn.Module) : initialized network
    """
    net = ResNetGenerator(in_channels=in_channels, out_channels=out_channels, num_blocks=num_blocks)
    return init_model(net, use_gpu=use_gpu, init_type=init_type, init_scale=init_scale)

def init_patch_discriminator(in_channels : int, num_filters : int = 64, num_conv_layers : int = 3, norm : str = 'instance', init_type : str = 'normal', init_scale : float = 0.02, use_gpu : bool = True):
    norm_layer = create_norm(norm)
    net = PatchDiscriminator(in_channels, num_filters, num_conv_layers, norm_layer)
    return init_model(net, use_gpu=use_gpu, init_type=init_type, init_scale=init_scale)

def init_linear_lr(optimizer : torch.optim, start_epoch : int, warmup_epochs : int, decay_epochs : int):
    """
    Simple linear learning rate scheduler.
    """
    def rule(epoch : int):
        return 1.0 - max(0.0, (start_epoch + epoch - warmup_epochs)) / float(decay_epochs)
    lr_schedule = lr_scheduler.LambdaLR(optimizer, rule)
    return lr_schedule
