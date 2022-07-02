import sys
sys.path.append('..')
from utils.str_to_layer import *

import torch.nn as nn

class ResNetBlock(nn.Module):
    """
    Class defining the building block of a residual (skip connection) convolutional network.
    """
    def __init__(self,
    dimension : int,
    dropout : bool,
    bias : bool,
    norm : str = 'instance',
    padding : str = 'reflect',
    kernel_size : int = 3):
        """
        Constructor for ResNetBlock.
            Parameters:
                dimension (int) : dimension of the convolutional layers
                dropuout (bool) : whether to use Dropout layers or not
                bias (bool) : whether to use bias in the convolutional layers or not
                norm (str) : normalization layer, default = 'instance'
                padding (str) : padding to use, helps reduce artifacts, default = 'reflect'
                kernel_size (int) : kernel size to use, default = 3
        """
        super(ResNetBlock, self).__init__()
        block = []

        # padding layer (if included at all)
        ones_padding, padding_layer = create_padding(padding)                
        pad_value = (1 if ones_padding else 0)

        if padding_layer != None: 
            block.append(padding_layer(1))

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

        if padding_layer != None:
            block.append(padding_layer(1))

        block.extend(
            [nn.Conv2d(dimension, dimension, kernel_size=kernel_size, padding=pad_value, bias=bias), 
            norm_layer(dimension)])

        self.res_block = nn.Sequential(*block)

    def forward(self, x):
        """
        Forward method for block. Adds identity skip connection.
            Parameters:
                x (tensor) : input into network
            Returns:
                output : output of network
        """
        output = self.res_block(x) + x # add skip connection
        return output
    
class ResNetGenerator(nn.Module):
    """
    Class defining a Residual-based Generator with multiple residual convolutional blocks.
    """
    def __init__(self,
    in_channels : int,
    out_channels : int,
    num_filters : int = 64,
    num_blocks : int = 6,
    num_sampling : int = 2,
    norm : str = 'instance',
    padding : str = 'reflect',
    use_dropout : bool = False):
        """
        Constructor for Residual Generator. 
            Parameters:
                in_channels (int) : number of channels in the input
                out_channels (int) : number of channels in the output
                num_filters (int) : number of filters in initial and last Convolutional layer, default = 64
                num_blocks (int) : number of ResNetBlocks to use, default = 6
                num_sampling (int) : number of intermediate downsampling and upsampling layers, default = 2
                norm (str) : normalization layer, default = 'instance'
                padding (str) : padding layer to use, default = 'reflect'
                use_dropout (bool) : whether to use Dropout layers in the ResNetBlocks
        """
        super(ResNetGenerator, self).__init__()
        # create norm layer
        norm_layer = create_norm(norm)
        use_bias = norm_layer == nn.InstanceNorm2d

        # 7x7 conv-instance-relu w/ 64 filters
        block = [nn.ReflectionPad2d(3),
        nn.Conv2d(in_channels, num_filters, kernel_size=7, stride=1, padding=0, bias=use_bias),
        norm_layer(num_filters),
        nn.ReLU(True)]

        # add downsampling layers
        # kernel size 3
        down_in_features = num_filters
        down_out_features = num_filters
        for _ in range(num_sampling):
            down_out_features *= 2
            block += [nn.Conv2d(down_in_features, down_out_features, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(down_out_features), 
            nn.ReLU(True)]
            down_in_features = down_out_features
            
        # add ResBlocks
        block_dimension = down_out_features
        for _ in range(num_blocks):
            block.append(ResNetBlock(block_dimension, dropout=use_dropout, padding=padding, bias=use_bias))

        # add upscaling layers
        # kernel size 3
        up_in_features = block_dimension
        up_out_features = block_dimension
        for _ in range(num_sampling):
            up_out_features = int(up_out_features/2)
            block += [nn.ConvTranspose2d(up_in_features, up_out_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(int(up_out_features)), 
            nn.ReLU(True)]
            up_in_features = up_out_features

        # add final layers, output on Tanh
        block += [nn.ReflectionPad2d(3),
        nn.Conv2d(num_filters, out_channels, kernel_size=7, padding=0),
        nn.Tanh()]

        self.model = nn.Sequential(*block)
    
    def forward(self, x):
        """
        Forward method for Residual Generator network.
            Parameters:
                x (tensor) : input into network
            Output: 
                output : output of the network
        """
        output = self.model(x)
        return output

class PatchDiscriminator(nn.Module):
    """
    Class to define a PatchGAN discriminator.
    PatchGAN is a discriminator architecture that outputs a value for each "patch" on the image
    convolutionally rather than one value for the entire image.
    """
    def __init__(self,
    num_channels : int,
    num_filters : int = 64,
    num_conv_layers : int = 3,
    norm_layer : nn.Module = nn.InstanceNorm2d,
    ker_size : int = 4,
    padding : int = 1):
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
                nn.LeakyReLU(0.2, True)]
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
#testNet = ResNetBlock(64, dropout=False, bias=True, norm='instance')
#print(testNet)

#testD = PatchDiscriminator(3)
#print(testD)

#testG = ResNetGenerator(3, 3)
#print(testG)

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
