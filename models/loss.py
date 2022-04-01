"""
| ||

|| |_
"""
import torch.nn as nn
import torch

class Loss(nn.Module):
    """
    Set up callable Loss class.
    """
    def __init__(self, type = 'mse', real_label = 1.0, fake_label = 0.0):
        """
        Constructor for Loss class.
            Parameters:
                type (str) : type of loss to use, default is mean squared loss
                real_label (bool) : real label
                fake_label (bool) : fake label
        """
        super(Loss, self).__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.type = type
        if type == 'mse':
            self.loss = nn.MSELoss()
        elif type == 'bce':
            self.loss = nn.BCEWithLogitsLoss()

    def get_labels(self, size : int, real : bool):
        """
        Sets up the labels. This is the key method, as it makes training simpler because we don't
        have to worry about setting up labels of the right size every time we call Loss.
        Additionally, we can easily expand this to implement soft labels/random label swapping, etc.
        Idea taken from original CycleGAN repo.
            Parameters:
                size (int) : size of tensor to construct
                real (bool) : whether to use real or fake labels defined in __init__ (true = real labels)
            Returns:
                (Tensor) : created tensor of labels
        """
        if real:
            return self.real_label.expand(size)
        return self.fake_label.expand(size)

    def __call__(self, predicted : torch.Tensor, real : bool):
        """
        Method that is executed when the class is called.
        Computes the loss between the predicted and the labels.
            Parameters:
                predicted (Tensor) : predicted values
                real (bool) : use real or fake labels
            Returns:
                loss (float) : float value of loss to use in backpropagation
        """
        labels = self.get_labels(predicted.size(dim=0), real)
        loss = self.loss(predicted, labels)
        return loss
