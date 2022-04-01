import torch

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

def list_equals(l1 : list, l2 : list) -> bool:
    if len(l1) != len(l2):
        return False
    for i in range(0, len(l1)):
        if l1[i] != l2[i]:
            return False
    return True