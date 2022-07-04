import unittest
from comparisons import list_equals, twoD_tensor_equals
import sys
sys.path.append('..')
from models.buffer import Buffer
import torch

class TestBuffer(unittest.TestCase):
    """
    Test suite for the Buffer class.
    """
    def test_init(self):
        """
        Test constructor.
        """
        buffer1 = Buffer(5)
        self.assertTrue(list_equals(buffer1.buffer, []))
        self.assertEqual(buffer1.buffer_size, 5)
        self.assertEqual(buffer1.num_tensors, 0)
    
    def test_query_edge(self):
        """
        Test edge cases of querying.
        """
        bufferzero = Buffer(0)
        bufferneg = Buffer(-1)
        to_add = [torch.Tensor([5]), torch.Tensor([5]), torch.Tensor([5]), torch.Tensor([5])]
        self.assertTrue(list_equals(bufferzero.query(to_add), to_add))
        self.assertTrue(list_equals(bufferneg.query(to_add), to_add))

    def test_query(self):
        """
        Test basic cases of querying to verify basic functionality (including randomness checks).
        """
        buffer = Buffer(2)
        to_add = [torch.Tensor([1]), torch.Tensor([1])]
        _ = buffer.query(to_add)
        to_overwrite = [torch.Tensor([2]), torch.Tensor([2])]
        returned = buffer.query(to_overwrite)

        self.assertEqual(2, len(returned)) # check property of size
        self.assertTrue(twoD_tensor_equals(returned, torch.Tensor([[1], [1]])) or
        twoD_tensor_equals(returned, torch.Tensor([[1], [2]])) or
        twoD_tensor_equals(returned, torch.Tensor([[2], [1]])) or
        twoD_tensor_equals(returned, torch.Tensor([[2], [2]])))

if __name__ == '__main__':
    unittest.main()
