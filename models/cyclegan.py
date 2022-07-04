from argparse import Namespace
import sys
sys.path.append('..')

import itertools
import os
import torch
import torch.nn as nn
from models.loss import Loss
from models.buffer import Buffer
from models.init_nets import init_linear_lr, init_resnet_generator, init_patch_discriminator
from utils.model_utils import print_network
from collections import OrderedDict

# average workflow:
# init network using options
# set up schedulers, should probably also print networks
# if continue train, load the networks
# forward thru network
# save networks whenever necessary
class CycleGAN(nn.Module):
    def __init__(self, opt : Namespace, config : dict):
        """
            Parameters:
                opt (Namespace) - options for instantiation
                config (dict) - config dictionary specifying the network architectures, probably parsed from a yaml
        """
        super(CycleGAN, self).__init__()
        self.opt = opt
        self.config = config
        self.to_train = opt.to_train

        # create log dir
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.model_name)
        print(f"Saving logs in {self.save_dir}")
        os.makedirs(self.save_dir, exist_ok=True)
        if opt.use_gpu:
            self.device = torch.device(('cuda:0'))
            torch.cuda.set_device(0)
        else:
            self.device = torch.device('cpu')

        # define nets
        self.in_channels = config['dataset']['in_channels']
        self.out_channels = config['dataset']['out_channels']
        self.genG = init_resnet_generator(self.in_channels, self.out_channels, use_gpu=opt.use_gpu, **config['model']['generator'])
        self.genF = init_resnet_generator(self.out_channels, self.in_channels, use_gpu=opt.use_gpu, **config['model']['generator'])
        self.model_names = ['genG', 'genF']
        if self.to_train:
            # only define this other stuff if we are planning on training the model
            # discriminator for X (D_X)
            self.netD_X = init_patch_discriminator(self.in_channels, **config['model']['discriminator'], use_gpu=opt.use_gpu)
            # discriminator for Y (D_Y)
            self.netD_Y = init_patch_discriminator(self.out_channels, **config['model']['discriminator'], use_gpu=opt.use_gpu) 
            self.model_names.append('netD_X')
            self.model_names.append('netD_Y')
        
            # define loss functions/objective functions
            self.loss_func = Loss(config['train']['loss']['loss_type']).to(self.device)
            self.loss_cycle = torch.nn.L1Loss()
            # loss for use with lambdas
            self.loss_identity = torch.nn.L1Loss()

            # define image pools
            self.buffer_size = config['train']['buffer_size']
            self.fake_X_buffer = Buffer(self.buffer_size)
            self.fake_Y_buffer = Buffer(self.buffer_size)
            
            # define optimizers and learning rate schedulers
            lr = config['train']['lr']
            beta1 = config['train']['beta1']
            self.optim_G = torch.optim.Adam(itertools.chain(self.genG.parameters(), self.genF.parameters()), lr=lr, betas=(beta1, 0.999))
            self.optim_D = torch.optim.Adam(itertools.chain(self.netD_X.parameters(), self.netD_Y.parameters()), lr=lr, betas=(beta1, 0.999))
            self.optimizers = [self.optim_G, self.optim_D]

    def setup_input(self, input : dict):
        """
        Sets the input in preparation for forwarding through the networks.
            Parameters:
                input (dict) : input dictionary
        """
        self.real_X = input['X'].to(self.device)
        self.real_Y = input['Y'].to(self.device)
        self.image_paths = input['X_paths']
    
    def forward(self):
        """
        Runs generator on real datasets and on the newly generated fake datasets.
        """
        # run generators on both real datasets
        self.fake_Y = self.genG(self.real_X)
        self.fake_X = self.genF(self.real_Y)
        # run generator on both fake generated sets
        self.fake_fake_X = self.genF(self.fake_Y) # F(G(X)), in domain X
        self.fake_fake_Y = self.genG(self.fake_X) # G(F(Y)), in domain Y
    
    def backward_D(self, discriminator : nn.Module, real : torch.Tensor, fake : torch.Tensor, factor : float = 0.5):
        """
        Calculate the backward pass for an arbitrary discriminator.        
            Parameters:
                discriminator (nn.Module) : discriminator to calculate loss for
                real (Tensor) : tensor of real images
                fake (Tensor) : tensor of fake images
                factor (float) : factor to scale loss by (can be tweaked, default 0.5 as that is the original CycleGAN paper implementation)
            Returns:
                loss : calculated loss gradients
        """
        # run on real dataset
        predict_real = discriminator(real.detach())
        loss_real = self.loss_func(predict_real, True)
        # run on fake generated (detach from device)
        predict_fake = discriminator(fake.detach())
        loss_fake = self.loss_func(predict_fake, False)
        # sum and calc gradient
        total_loss = (loss_real + loss_fake) * factor
        total_loss.backward()
        return total_loss

    def backward_D_X(self):
        """ Calculate gradients for discriminator on domain X """
        fake_X = self.fake_X_buffer.query(self.fake_X)
        self.loss_D_X = self.backward_D(self.netD_X, self.real_X, fake_X)

    def backward_D_Y(self):
        """ Calculate gradients for discriminator on domain Y """
        fake_Y = self.fake_Y_buffer.query(self.fake_Y) 
        self.loss_D_Y = self.backward_D(self.netD_Y, self.real_Y, fake_Y)
    
    def backward_G(self):
        """
        Calculates the total loss for the generators and calls .backward().
        Currently uses discriminator loss, cycle loss, and identity loss if lambda_scaling is present.
        """
        # use options
        lambda_scaling = self.config['train']['loss']['lambda_scaling']
        lambda_X = self.config['train']['loss']['lambda_X']
        lambda_Y = self.config['train']['loss']['lambda_Y']

        # implement lambda scaled loss in the future, but for now use 0
        # need something here for identity loss
        # lambda/identity loss of G_X(y) --> G_X should mimic the identity map
        if lambda_scaling > 0:
            identity_output_Y = self.genG(self.real_Y) # lives in Y
            self.identity_loss_X = self.loss_identity(identity_output_Y, self.real_Y) * lambda_scaling * lambda_Y
            identity_output_X = self.genF(self.real_X) # lives in X
            self.identity_loss_Y = self.loss_identity(identity_output_X, self.real_X) * lambda_scaling * lambda_X
        else:
            self.identity_loss_X = 0
            self.identity_loss_Y = 0

        # calculate loss of D_X(F(y)) [X domain fakes]
        # compare against real labels because we want D to think they are real
        self.loss_domain_X = self.loss_func(self.netD_X(self.fake_X), True)
        # loss of D_Y(G(x)) [Y domain fakes]
        self.loss_domain_Y = self.loss_func(self.netD_Y(self.fake_Y), True)
        # cycle loss of ||G(F(Y)) - Y||
        self.cycle_loss_Y = self.loss_cycle(self.fake_fake_Y, self.real_Y) * lambda_Y # G(F) lives in Y
        # cycle loss of ||F(G(X)) - X||
        self.cycle_loss_X = self.loss_cycle(self.fake_fake_X, self.real_X) * lambda_X # F(G) lives in X

        self.loss_G = self.loss_domain_X + self.loss_domain_Y + self.cycle_loss_X + self.cycle_loss_Y + self.identity_loss_X + self.identity_loss_Y
        self.loss_G.backward()
    
    def optimize(self):
        """
        Main method called every training loop. Forwards inputs and backpropagates losses.
        """
        # forward thru network
        self.forward()

        # optimize generators
        self.optim_G.zero_grad()
        self.backward_G()
        self.optim_G.step()

        # optimize discriminators
        self.optim_D.zero_grad()
        self.backward_D_X()
        self.backward_D_Y()
        self.optim_D.step()

    # utils
    def general_setup(self):
        """
        Sets up model's networks and optimizers according to whether it is being run for training or not.
        Prints networks with verbose option at end.
        """
        if self.to_train:
            self.setup_schedulers()
        if not self.to_train or self.opt.continue_train:
            self.load_networks(self.opt.load_epoch)
        
        print_network(self, self.opt.verbose)

    def setup_schedulers(self):
        """
        Sets up the list of learning rate schedulers for all optimizers.
        """
        self.schedulers = [init_linear_lr(optimizer, **self.config['train']) for optimizer in self.optimizers]
        
    def update_schedulers(self):
        """
        Updates all schedulers.
        """
        for scheduler in self.schedulers:
            scheduler.step()

    def save_networks(self, epoch : str = 'latest'):
        """
        Saves network state dictionaries in a .pth file.
            Parameters:
                epoch (str) : which epoch these models are from. 
                    used later on when loading the models
        """
        for model_name in self.model_names:
            filename = f'{epoch}_{model_name}.pth'
            save_path = os.path.join(self.save_dir, filename)
            net = getattr(self, model_name)
            torch.save(net.state_dict(), save_path)

    def load_networks(self, epoch : str = 'latest'):
        """
        Loads the network specified by epoch.
            Parameters:
                epoch (str) : which epoch to load, default is 'latest'
        """
        for model_name in self.model_names:
            load_path = os.path.join(self.save_dir, f'{epoch}_{model_name}.pth')
            net = getattr(self, model_name)
            state_dict = torch.load(load_path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)
        
    def eval(self):
        """
        Sets models in evaluation mode (turns off gradients, etc.) to avoid unnecessary computations.
        """
        for model_name in self.model_names:
            net = getattr(self, model_name)
            net.eval()
    
    def get_losses(self):
        """
        Must be called after all the values in the dict have been created.
            (This is satisfied after one call of optimize())
        """
        return OrderedDict(
           D_X=self.loss_D_X,
           D_Y=self.loss_D_Y,
           G=self.loss_domain_Y,
           F=self.loss_domain_X,
           cycle_X=self.cycle_loss_X,
           cycle_Y=self.cycle_loss_Y, 
           identity_X=self.identity_loss_X,
           identity_Y=self.identity_loss_Y,
        )

