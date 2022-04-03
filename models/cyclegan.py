import torch
import torch.nn as nn
from init_nets import init_resnet_generator, init_patch_discriminator
from loss import Loss

class CycleGAN(nn.Module):
    def __init__(self, loss_type : str = 'mse', use_gpu : bool = True, to_train : bool = True):
        self.device = torch.device(('cuda:0') if use_gpu else "cpu")

        # define nets
        self.genG = init_resnet_generator()  # runs from domain X to Y
        self.genF = init_resnet_generator() # runs from domain Y to X
        if to_train:
            self.netD_X = init_patch_discriminator() # discriminator for domain X
            self.netD_Y = init_patch_discriminator() # discriminator for domain Y
        
            # define loss functions/objective functions
            self.loss_func = Loss(loss_type=loss_type).to(self.device)
            self.loss_cycle = torch.nn.L1Loss()
            # loss for use with lambdas
            self.loss_identity = torch.nn.L1Loss()

            # define image pools
            self.fake_X_pool = ()
            self.fake_Y_pool = ()
            
            # define optimizers

    def setup_input(self, input : dict):
        self.real_X = input['X'].to(self.device)
        self.real_Y = input['Y'].to(self.device)
    
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
        fake_X = self.fake_X_pool.query(self.fake_X)
        self.loss_D_X = self.backward_D(self.netD_X, self.real_X, fake_X)

    def backward_D_Y(self):
        """ Calculate gradients for discriminator on domain Y"""
        fake_Y = self.fake_Y_pool.query(self.fake_Y) 
        self.loss_D_Y = self.backward_D(self.netD_Y, self.real_Y, fake_Y)
    
    def backward_G(self):
        # use options
        lambda_X = self.opt.lambda_X
        lambda_Y = self.opt.lambda_Y

        # implement lambda scaled loss in the future, but for now use 0
        # need something here for identity loss
        # cycle losses should be multiplied, but currently not implemented (WIP)

        # calculate loss of D_X(F(y)) [X domain fakes]
        self.loss_domain_X = self.loss_func(self.netD_X(self.fake_X))
        # loss of D_Y(G(x)) [Y domain fakes]
        self.loss_domain_Y = self.loss_func(self.netD_Y(self.fake_Y))
        # cycle loss of D_Y(G(F(y)))
        self.cycle_loss_Y = self.loss_cycle(self.netD_Y(self.fake_fake_Y, self.real_Y)) * lambda_Y # G(F) lives in Y
        # cycle loss of D_X(F(G(x)))
        self.cycle_loss_X = self.loss_cycle(self.netD_X(self.fake_fake_X, self.real_X)) * lambda_X # F(G) lives in X

        self.loss_G = self.loss_domain_X + self.loss_domain_Y + self.cycle_loss_X + self.cycle_loss_Y
        self.loss_G.backward()