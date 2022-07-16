# set up import
import sys
sys.path.append('..')

import torch
import numpy as np
from models.loss import Loss
from comparisons import oned_tensor_equals

def test_get_labels():
    """
    Tests the get_labels method.
    """
    test = Loss()
    assert oned_tensor_equals(torch.Tensor(np.ones(64)), test.get_labels(torch.Tensor(np.empty(64)), True))
    assert oned_tensor_equals(torch.Tensor(np.zeros(64)), test.get_labels(torch.Tensor(np.empty(64)), False))

# test for computation

