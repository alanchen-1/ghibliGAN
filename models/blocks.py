from msilib.schema import Patch
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
    
# UNET Architecture
"""
class DownsampleBlock(nn.Module): 
    def __init__(self, dim : int, use_bias : bool):
        super(DownsampleBlock, self).__init__()

        self.block = self.create_block(dim, use_bias)
    def create_block(dim : int, use_bias : bool) :
        block = []
        block += [nn.Conv2d(dim, dim stride=2),
        nn.InstanceNorm2d()]
        # Conv2d layer, stride 2
        # padding = 1
        # norm = instancenorm
        # leaky relu
        return nn.Sequential(*block)
    
#    def forward(self, x):
"""

class PatchDiscriminator(nn.Module):
    """
    Class to define a PatchGAN discriminator.
    """
    def __init__(self, num_channels : int,
    num_filters : int=64, num_conv_layers : int=3, norm_layer=nn.InstanceNorm2d, ker_size : int=4, padding : int=1):
        """
        Constructor for PatchGAN discriminator.
            Parameters:
                num_channels (int) : number of channels in input (for RGB, num_channels = 3)
                num_filters (int) : number of filters in first convolutional layer, default is 64
                num_conv_layers (int) : number of convolutional layers, default is 3
                norm_layer (nn.Module) : normalization layer, default is nn.InstanceNorm2d
                ker_size (int) : kernel size to use for convolutional layers, default is 4
                padding (int) : padding value to use, default is 1
        """
        super(PatchDiscriminator, self).__init__()
        
        # only need bias if using InstanceNorm
        # idea taken from original CycleGAN repository
        bias = (norm_layer == nn.InstanceNorm2d) 

        block = [nn.Conv2d(num_channels, num_filters, kernel_size=ker_size, stride=2, padding=padding), 
        nn.LeakyReLU(0.2, True)]

        in_filters = num_filters
        out_filters = None
        for layer in range(1, num_conv_layers + 1):
            out_filters = min(in_filters * 2, num_filters * 8) # set upper limit on size of layers

            # on last layer, change stride to 1
            if layer < num_conv_layers:
                block.append(nn.Conv2d(in_filters, out_filters, kernel_size=ker_size, stride=2, padding=padding, bias=bias))
            else:
                block.append(nn.Conv2d(in_filters, out_filters, kernel_size=ker_size, stride=1, padding=padding, bias=bias))

            block += [
                norm_layer(out_filters),
                nn.LeakyReLU(0.2, True)
            ]
            in_filters = out_filters
        
        # output layer = 1 channel
        block.append(nn.Conv2d(out_filters, 1, kernel_size=ker_size, stride=1, padding=padding))
        self.model = nn.Sequential(*block)
    
    def forward(self, x):
        """
        Forward method for network.
            Parameters:
                x (tensor) : input
            Returns:
                output : value after x is forwarded through the network
        """
        output =  self.model(x)
        return output

# test stuff
testNet = ResNetBlock(64, dropout=False, bias=True, norm='instance')
print(testNet)

testD = PatchDiscriminator(3)
print(testD)
