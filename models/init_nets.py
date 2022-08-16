import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from models.blocks import ResNetGenerator, PatchDiscriminator
from utils.str_to_layer import create_norm


def init_params(
    network: nn.Module,
    init_type: str,
    init_scale: float = 0.02
) -> None:
    """
    Initializes the weights and biases of the network.
        Parameters:
            network (nn.Module) : network to initialize
            init_type (str) : initialization type to use
            init_scale (float) : initialization scale
                (variance for norm, gain for xavier)
    """
    def init_params_helper(m: nn.Module):
        """
        Helper to apply to the model.
            Parameters:
                model (nn.Module) : network/layer to apply to
        """
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
            classname.find('Conv') != -1 or classname.find('Linear') != -1
        ):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_scale)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_scale)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(
                    f'{type} initialization not implemented'
                )
            # init bias to 0 if has bias
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_scale)
            init.constant_(m.bias.data, 0.0)
    # apply to each layer in the network
    network.apply(init_params_helper)


def add_device(network: nn.Module, use_gpu: bool = True) -> nn.Module:
    """
    Attaches model to device to allow for GPU training.
    Currently only attaches to one GPU (since I only have one),
    but easily expandable to incorporate multiple GPU support
    using Data.Parallel (maybe a future update)
        Parameters:
            network (nn.Module) : network to move to a device
            use_gpu (bool) : whether to use GPU (if available, default is True)
        Returns:
            network (nn.Module) : original network attached to the
                specified device
    """
    device = torch.device("cuda:0" if (
        use_gpu and torch.cuda.is_available()
    ) else "cpu")
    network.to(device)
    if use_gpu:
        return torch.nn.DataParallel(network, [0])
    else:
        return network


def init_model(
    network: nn.Module,
    use_gpu: bool = True,
    init_type: str = 'normal',
    init_scale: float = 0.02
) -> nn.Module:
    """
    Initializes a model by adding it to a device and initializing the weights.
        Parameters:
            network (nn.Module) : network to initialize
            use_gpu (bool) : whether to use GPU in training
            init_type (str) : initialization type
            init_scale (float) : initialization scale,
                only applicable for normal or xavier
        Returns:
            (nn.Module) : initialized network
    """
    network = add_device(network, use_gpu)
    init_params(network, init_type=init_type, init_scale=init_scale)
    return network


def init_resnet_generator(
    in_channels: int,
    out_channels: int,
    use_gpu: bool,
    num_filters: int,
    num_blocks: int,
    norm: str = 'instance',
    num_sampling: int = 2,
    init_type: str = 'normal',
    init_scale: float = 0.02,
    use_dropout: bool = False,
    **kwargs
):
    """
    Define and initialize a simple ResNetGenerator
    (uses a lot of default values).
    Provides a template to define custom initalize methods like this.
        Parameters:
            in_channels (int) : channels in inputs
            out_channels (int) : channels in outputs
            num_blocks (int) : number of ResNetBlocks
            num_filters (int) : number of filters in first convolutional layer
            norm (str) : normalization layers to use (after convs)
            use_gpu (bool) : whether to use GPU in training
            init_type (str) : initialization type
            init_scale (float) : initialization scale,
                only applicable for normal or xavier
            use_dropout (bool) : whether to use dropout layers
        Returns:
            (nn.Module) : initialized Generator
    """
    net = ResNetGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        num_filters=num_filters,
        num_blocks=num_blocks,
        norm=norm,
        use_dropout=use_dropout,
        num_sampling=num_sampling
    )
    return init_model(
        net, use_gpu=use_gpu, init_type=init_type, init_scale=init_scale
    )


def init_patch_discriminator(
    in_channels: int,
    num_filters: int = 64,
    num_conv_layers: int = 3,
    norm: str = 'instance',
    ker_size: int = 4,
    padding: int = 1,
    init_type: str = 'normal',
    init_scale: float = 0.02,
    use_gpu: bool = True
):
    """
    Define and initialize a Patch Discriminator.
        Parameters:
            in_channels (int) : channels in input
            num_filters (int) : number of filters in last
                conv layer
            num_conv_layers (int) : number of
                convolutional-instance-leakyrelu blocks
            norm (str) : normalization layers to use
            init_type (str) : initialization type
            init_scale (float) : initialization scale
                (variance for norm, gain for xavier)
            use_gpu (bool) : whether to use GPU
        Returns:
            (nn.Module) : initialized Discriminator
    """
    norm_layer = create_norm(norm)
    net = PatchDiscriminator(
        in_channels, num_filters, num_conv_layers, norm_layer, ker_size, padding
    )
    return init_model(
        net, use_gpu=use_gpu, init_type=init_type, init_scale=init_scale
    )


def init_linear_lr(
    optimizer: torch.optim,
    start_epoch: int,
    warmup_epochs: int,
    decay_epochs: int,
    **kwargs
):
    """
    Simple linear learning rate scheduler.
    Holds lr constant for <warmup_epochs>, then linearly decays to 0
    over the next <decay_epochs>.
        Parameters:
            optimizer (torch.optim) : optimizer to add LR to
            start_epoch (int) : starting epoch (used to continue
                training a model)
            warmup_epochs (int) : number of epochs to hold LR constant for
            decay_epochs (int) : number of epochs to decay LR to 0 for
        Returns:
            (lr_scheduler.LambdaLR) : learning rate scheduler
    """
    def multiplier(epoch: int):
        """
        Defines rule to use when calculating learning rate @ epoch <epoch>.
            Parameters:
                epoch (int) : current epoch
            Returns:
                (float) : learning rate multiplier at that epoch
                    (linearly decays)
        """
        # add one to make sure there is never an epoch with 0 lr
        lam = 1.0 - max(
            0.0, start_epoch + epoch - warmup_epochs
        ) / float(decay_epochs + 1)
        return lam
    lr_schedule = lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
    return lr_schedule
