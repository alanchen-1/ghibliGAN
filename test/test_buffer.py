# non deterministic test cases for image_buffer
import unittest
from test_utils import list_equals, twoD_tensor_equals
import sys
sys.path.append('..')
from models.buffer import Buffer
import torch

class TestBuffer(unittest.TestCase):
    def test_init(self):
        buffer1 = Buffer(5)
        self.assertTrue(list_equals(buffer1.buffer, []))
        self.assertEqual(buffer1.buffer_size, 5)
        self.assertEqual(buffer1.num_tensors, 0)
    
    def test_query_edge(self):
        bufferzero = Buffer(0)
        bufferneg = Buffer(0)
        to_add = [torch.Tensor([5]), torch.Tensor([5]), torch.Tensor([5]), torch.Tensor([5])]
        self.assertTrue(list_equals(bufferzero.query(to_add), to_add))
        self.assertTrue(list_equals(bufferneg.query(to_add), to_add))

    def test_query(self):
        buffer = Buffer(2)
        to_add = [torch.Tensor([1]), torch.Tensor([1])]
        _ = buffer.query(to_add)
        to_overwrite = [torch.Tensor([2]), torch.Tensor([2])]
        returned = buffer.query(to_overwrite)

        self.assertEqual(2, len(returned))
        self.assertTrue(twoD_tensor_equals(returned, torch.Tensor([[1], [1]])) or
        twoD_tensor_equals(returned, torch.Tensor([[1], [2]])) or
        twoD_tensor_equals(returned, torch.Tensor([[2], [1]])) or
        twoD_tensor_equals(returned, torch.Tensor([[2], [2]])))
        

if __name__ == '__main__':
    unittest.main()
