import torch
import numpy as np

from protstruc.decorator import with_tensor


@with_tensor
def mysum(a, b):
    return a + b


def test_with_tensor():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])

    x = mysum(a, b)

    assert isinstance(x, torch.Tensor)
    assert (x == torch.tensor([5, 7, 9])).all()


def test_with_numpy():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    x = mysum(a, b)

    assert isinstance(x, np.ndarray)
    assert (x == np.array([5, 7, 9])).all()


def test_mixed():
    a = torch.tensor([1, 2, 3])
    b = np.array([4, 5, 6])

    x = mysum(a, b)

    assert isinstance(x, torch.Tensor)
    assert (x == torch.tensor([5, 7, 9])).all()
