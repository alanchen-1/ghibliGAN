import sys
sys.path.append('..')

import itertools
import os
import torch
import torch.nn as nn
from loss import Loss
from buffer import Buffer
from models.init_nets import init_linear_lr, init_resnet_generator, init_patch_discriminator
from utils.model_utils import print_network

# average workflow:
# init network using options
# set up schedulers, should probably also print networks
# if continue train, load the networks
# forward thru network
# save networks whenever necessary
class CycleGAN(nn.Module):
    def __init__(self, opt):
        """
            Parameters:
                opt (Options) - options for instantiation
        """
        self.opt = opt
        self.to_train = opt.to_train
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.model_name)
        self.device = torch.device(('cuda:0') if opt.use_gpu else "cpu")

        # define nets
        self.genG = init_resnet_generator(opt.in_channels, opt.out_channels,
        opt.num_g_f, opt.num_g_blocks, opt.norm, opt.use_gpu,
        opt.init_type, opt.init_scale, opt.use_dropout)  # runs from domain X to Y
        self.genF = init_resnet_generator(opt.out_channels, opt.in_channels,
        opt.num_g_f, opt.num_g_blocks, opt.norm, opt.use_gpu, opt.init_type,
        opt.init_scale, opt.use_dropout)  # runs from domain Y to Y
        self.model_names = ['genG', 'genF']
        if opt.to_train:
            # only define this other stuff if we are planning on training the model
            # discriminator for X (D_X)
            self.netD_X = init_patch_discriminator(opt.in_channels, opt.num_d_f, opt.num_d_layers, opt.norm, opt.init_type, opt.init_scale, opt.use_gpu)
            # discriminator for Y (D_Y)
            self.netD_Y = init_patch_discriminator(opt.out_channels, opt.num_d_f, opt.num_d_layers, opt.norm, opt.init_type, opt.init_scale, opt.use_gpu) 
            self.model_names.append('netD_X')
            self.model_names.append('netD_Y')
        
            # define loss functions/objective functions
            self.loss_func = Loss(loss_type=opt.loss_type).to(self.device)
            self.loss_cycle = torch.nn.L1Loss()
            # loss for use with lambdas
            self.loss_identity = torch.nn.L1Loss()

            # define image pools
            self.fake_X_buffer = Buffer(opt.buffer_size)
            self.fake_Y_buffer = Buffer(opt.buffer_size)
            
            # define optimizers and learning rate schedulers
            self.optim_G = torch.optim.Adam(itertools.chain(self.genG.parameters(), self.genF.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optim_D = torch.optim.Adam(itertools.chain(self.netD_X.parameters(), self.netD_Y.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optim_G, self.optim_D]

    def setup_input(self, input : dict):
        self.real_X = input['X'].to(self.device)
        self.real_Y = input['Y'].to(self.device)
        self.image_paths = input['X_paths']
    
    def forward(self):
        # run generators on both real datasets
        self.fake_Y = self.genG(self.real_X)
        self.fake_X = self.genF(self.real_Y)
        # run generator on both fake generated sets
        self.fake_fake_X = self.genF(self.fake_Y)
        self.fake_fake_Y = self.genG(self.fake_X)
    
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
        predict_real = discriminator(real)
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
        """ Calculate gradients for discriminator on domain Y"""
        fake_Y = self.fake_Y_buffer.query(self.fake_Y) 
        self.loss_D_Y = self.backward_D(self.netD_Y, self.real_Y, fake_Y)
    
    def backward_G(self):
        # use options
        lambda_scaling = self.opt.lambda_scaling
        lambda_X = self.opt.lambda_X
        lambda_Y = self.opt.lambda_Y

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
        # cycle loss of D_Y(G(F(y)))
        self.cycle_loss_Y = self.loss_cycle(self.netD_Y(self.fake_fake_Y, self.real_Y)) * lambda_Y # G(F) lives in Y
        # cycle loss of D_X(F(G(x)))
        self.cycle_loss_X = self.loss_cycle(self.netD_X(self.fake_fake_X, self.real_X)) * lambda_X # F(G) lives in X

        self.loss_G = self.loss_domain_X + self.loss_domain_Y + self.cycle_loss_X + self.cycle_loss_Y + self.identity_loss_X + self.identity_loss_Y
        self.loss_G.backward()
    
    def optimize(self):
        # do everything above!    
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
    def general_setup(self, opt):
        if self.to_train:
            self.setup_schedulers(opt)
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.load_epoch)
        
        print_network(opt.verbose)


    def setup_schedulers(self, opt):
        self.schedulers = [init_linear_lr(optimizer, opt.start_epoch, opt.warmup_epochs, opt.decay_epochs) for optimizer in self.optimizers]
        
    def update_schedulers(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def save_networks(self, epoch : str):
        for model_name in self.model_names:
            filename = f'{epoch}_{model_name}.pth'
            save_path = os.path.join(self.save_dir, filename)
            net = getattr(self, model_name)
            torch.save(net.state_dict(), save_path)

    def load_networks(self, epoch : str):
        for model_name in self.model_names:
            load_path = os.path.join(self.save_dir, f'{epoch}_{model_name}.pth')
            net = getattr(self, model_name)
            state_dict = torch.load(load_path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)
        


