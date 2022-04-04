import torch

def oned_tensor_equals(t1 : torch.Tensor, t2 : torch.Tensor) -> bool:
    """
    Tests if two one dimensional tensors are elementwise equal.
        Parameters:
            t1 (Tensor) : first tensor
            t2 (Tensor) : second tensor
        Returns:
            (bool) : if they are elementwise equal
    """
    if t1.size(dim=0) != t2.size(dim=0):
        return False
    for i in range(0, t1.size(dim=0)):
        if t1[i] != t2[i]:
            return False
    return True

def twoD_tensor_equals(t1 : torch.Tensor, t2 : torch.Tensor) -> bool:
    """
    Tests if two two dimensional tensors are elementwise equal.
        Parameters:
            t1 (Tensor) : first tensor
            t2 (Tensor) : second tensor
        Returns:
            (bool) : if they are elementwise equal
    """
    if t1.size(dim=0) != t2.size(dim=0) or t1.size(dim=1) != t2.size(dim=1):
        return False
    for i in range(0, t1.size(dim=0)):
        for j in range(0, t1.size(dim=1)):
            if t1[i][j] != t2[i][j]:
                return False
    return True
    
def list_equals(l1 : list, l2 : list) -> bool:
    """
    Tests if two lists are equivalent elementwise.
        Parameters:
            l1 (list) : first list
            l2 (list) : second list
        Returns:
            (bool) : if they are equal
    """
    if len(l1) != len(l2):
        return False
    for i in range(0, len(l1)):
        if l1[i] != l2[i]:
            return False
    return True
