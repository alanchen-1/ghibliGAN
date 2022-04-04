# implements tensor buffer for discriminator, cyclegan uses size 50 of 2d tensors (images)
import random
import torch

class Buffer():
    def __init__(self, buffer_size : int):
        self.buffer_size = buffer_size
        self.buffer = []
        self.num_tensors = 0
    
    def query(self, fake_tensors : list[torch.Tensor]):
        if self.buffer_size <= 0:
            return fake_tensors
        
        to_return = []
        for fake_tensor in fake_tensors:
            fake_tensor = torch.unsqueeze(fake_tensor.data, 0)
            if self.num_tensors < self.buffer_size:
                self.num_tensors += 1
                self.buffer.append(fake_tensor)
                to_return.append(fake_tensor)
                continue

            # select random
            p = random.random()
            if p < 0.5:
                # add random real tensor
                index = random.randint(0, self.num_tensors - 1)
                to_return.append(self.buffer[index].clone())
                self.buffer[index] = fake_tensor
            else:
                # add fake image
                to_return.append(fake_tensor)
            
        return torch.cat(to_return, 0)
