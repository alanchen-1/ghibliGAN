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
    def __init__(
        self,
        loss_type: str = 'mse',
        targ_real_label: float = 1.0,
        targ_fake_label: float = 0.0
    ):
        """
        Constructor for Loss class.
            Parameters:
                type (str) : type of loss to use, default is mean squared loss
                real_label (bool) : real label
                fake_label (bool) : fake label
        """
        super(Loss, self).__init__()
        self.register_buffer('real_label', torch.tensor(targ_real_label))
        self.register_buffer('fake_label', torch.tensor(targ_fake_label))
        self.loss_type = loss_type
        if loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif loss_type == 'bce':
            self.loss = nn.BCEWithLogitsLoss()

    def get_labels(self, predicted: torch.Tensor, real: bool):
        """
        Sets up the labels. This is the key method, as it makes
        training simpler because we don't have to worry about setting
        up labels of the right size every time we call Loss. Additionally,
        we can easily expand this to implement soft labels/random
        label swapping, etc.
        Idea taken from original CycleGAN repo.
            Parameters:
                size (int) : size of tensor to construct
                real (bool) : whether to use real or fake labels defined in
                    __init__ (true = real labels)
            Returns:
                (Tensor) : created tensor of labels
        """
        if real:
            return self.real_label.expand_as(predicted)
        return self.fake_label.expand_as(predicted)

    def __call__(self, predicted: torch.Tensor, real: bool):
        """
        Method that is executed when the class is called.
        Computes the loss between the predicted and the labels.
            Parameters:
                predicted (Tensor) : predicted values
                real (bool) : use real or fake labels
            Returns:
                loss (float) : float value of loss to use in backpropagation
        """
        labels = self.get_labels(predicted, real)
        if predicted.shape[0] == 0 and self.loss_type == 'mse':
            raise ZeroDivisionError(
                "Length of predicted tensor is not allowed to be 0 in MSE"
            )
        loss = self.loss(predicted, labels)
        return loss
