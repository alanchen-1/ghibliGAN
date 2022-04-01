# set up import
import sys
sys.path.append('..')

import unittest
import torch
import numpy as np
from models.loss import Loss
from test_utils import oned_tensor_equals

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

