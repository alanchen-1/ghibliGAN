import torch
import torch.nn as nn


def create_norm(norm : str):
    if norm == 'instance':
        return nn.InstanceNorm2d
    elif norm == 'batch':
        return nn.BatchNorm2d

def create_padding(padding : str):
    if padding == 'reflect':
        return True, nn.ReflectionPad2d
    else:
        return False, None

class ResNetBlock(nn.Module):
    def __init__(self,
    dimension : int,
    dropout : bool,
    bias : bool,
    norm : str = 'instance',
    padding : str = 'reflect',
    kernel_size : int = 3):
        super(ResNetBlock, self).__init__()
        self.main_block = self.construct_block(dimension, dropout, bias, norm, padding, kernel_size)

    def construct_block(self, 
    dimension : int,
    dropout : bool,
    bias : bool,
    norm : str,
    padding : str,
    kernel_size : int) -> nn.ModuleList():
        block = nn.ModuleList()

        # padding layer (if included at all)
        has_padding, padding_layer = create_padding(padding)                
        if has_padding:
            block.append(padding_layer(1))
        pad_value = (1 if has_padding else 0)

        # norm layer
        norm_layer = create_norm(norm)

        # add layers
        block.extend(
            [nn.Conv2d(dimension, dimension, kernel_size=kernel_size, padding=pad_value, bias=bias), 
            norm_layer(dimension), 
            nn.ReLU(True)])

        # add dropout if requested
        if dropout:
            block.append(nn.Dropout(p=0.5))

        if has_padding:
            block.append(padding_layer(1))

        block.extend(
            [nn.Conv2d(dimension, dimension, kernel_size=kernel_size, padding=pad_value, bias=bias), 
            norm_layer(dimension)])

        return block

    
    def forward(self, x):
        output = x
        for layer in self.main_block:
            output = layer(output)
        output += x # add skip connection
        return output
    

# test stuff
testNet = ResNetBlock(64, dropout=False, bias=True, norm='instance')
print(testNet)
