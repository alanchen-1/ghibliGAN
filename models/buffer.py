# implements tensor buffer for discriminator
# cyclegan uses size 50 of 2d tensors (images)
import random
import torch


class Buffer():
    """
    Class that implements tensor buffer for the discriminator to use.
    Original paper uses a size 50 buffer.
    """
    def __init__(self, buffer_size: int):
        """
        Constructor for Buffer class.
            Parameters:
                buffer_size (int) : size of the buffer (will not exceed this)
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.num_tensors = 0

    def query(self, fake_tensors: "list[torch.Tensor]"):
        """
        Queries the buffer.
            Parameters:
                fake_tensors (list[torch.Tensor]) : list
                    of fake tensors used for the query
            Returns:
                (torch.Tensor) : tensor of the queried tensors.
                    non-deterministic due to random selection of images to add.
        """
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
            p = random.uniform(0, 1)
            print(p)
            if p < 0.5:
                # add random real tensor and replace it in the buffer so that
                # D still only sees each image once
                index = random.randint(0, self.num_tensors - 1)
                to_return.append(self.buffer[index].clone())
                self.buffer[index] = fake_tensor
            else:
                # add fake image
                to_return.append(fake_tensor)

        return torch.cat(to_return, 0)
