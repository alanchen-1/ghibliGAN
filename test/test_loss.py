# set up import
import sys
sys.path.append('..')

import torch
import numpy as np
from models.loss import Loss
import pytest


def test_get_labels():
    """
    Tests the get_labels method.
    """
    test = Loss()
    assert torch.equal(
        torch.Tensor(np.ones(64)),
        test.get_labels(torch.Tensor(np.empty(64)), True))
    assert torch.equal(
        torch.Tensor(np.zeros(64)),
        test.get_labels(torch.Tensor(np.empty(64)), False))


def test_mse():
    """
    Tests mse.
    """
    test = Loss(loss_type='mse')

    predicted = torch.Tensor([1, 0, 2, 2])
    assert pytest.approx(0.75) == test(predicted, True)

    assert 0 == test(torch.Tensor(np.zeros(50)), False)
    with pytest.raises(ZeroDivisionError):
        test(torch.Tensor([]), False)
        test(torch.Tensor([]), True)
