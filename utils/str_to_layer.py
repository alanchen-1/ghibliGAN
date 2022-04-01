import torch.nn as nn
def create_norm(norm : str):
    if norm == 'instance':
        return nn.InstanceNorm2d
    elif norm == 'batch':
        return nn.BatchNorm2d

def create_padding(padding : str):
    if padding == 'reflect':
        return False, nn.ReflectionPad2d
    elif padding == 'ones':
        return True, None
    else:
        return False, None