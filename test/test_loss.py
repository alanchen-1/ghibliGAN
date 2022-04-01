# set up import
import sys
sys.path.append('..')

import unittest
import torch
import numpy as np
from models.loss import Loss

def oned_tensor_equals(t1 : torch.Tensor, t2 : torch.Tensor) -> bool:
    """
    Tests if two one dimensional tensors are elementwise equal.
        Parameters:
            t1 (Tensor) : first tensor
            t2 (Tensor) : second tensor
        Returns:
            (boolean) : if they are elementwise equal
    """
    if t1.size(dim=0) != t2.size(dim=0):
        return False
    for i in range(0, t1.size(dim=0)):
        if t1[i] != t2[i]:
            return False
    return True

class TestLoss(unittest.TestCase):
    """
    Test cases for Loss class.
    """
    def test_get_labels(self):
        """
        Tests the get_labels method.
        """
        test = Loss()
        self.assertTrue(oned_tensor_equals(torch.Tensor(np.ones(64)), test.get_labels(64, True)))
        self.assertTrue(oned_tensor_equals(torch.Tensor(np.zeros(64)), test.get_labels(64, False)))

if __name__ == '__main__':
    unittest.main()

