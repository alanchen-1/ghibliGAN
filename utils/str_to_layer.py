import torch.nn as nn


def create_norm(norm: str):
    """
    Converts a norm string/name to a nn.Module layer.
        Parameters:
            norm (str) : name of norm layer
        Returns:
            (nn.Module) : corresponding layer
        Raises:
            (NotImplementedError) : if name is not recognized
    """
    if norm == 'instance':
        return nn.InstanceNorm2d
    elif norm == 'batch':
        return nn.BatchNorm2d
    else:
        raise NotImplementedError(f'{norm} normalization not supported')


def create_padding(padding: str):
    """
    Converts a padding to a layer.
        Parameters:
            padding (str) : padding layer name
        Returns:
            (bool) : is constant padding
            (nn.Module) : corresponding layer
    """
    if padding == 'reflect':
        return False, nn.ReflectionPad2d
    elif padding == 'ones':
        return True, None
    else:
        return False, None
