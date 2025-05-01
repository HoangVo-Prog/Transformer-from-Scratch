import pytest
import torch
from utils import subsequent_mask, clones
from model import Transformer


def test_subsequent_mask():
    size = torch.randint(0, 10, (1,)).item()
    
    mask = subsequent_mask(size)
    for i in range(size):
        for j in range(size):
           assert mask[0, i, j] == (1 if j <= i else 0), f"Failed for size {size}: expected mask[{i}, {j}] to be {(1 if j <= i else 0)}"
         

def test_clones():
    module = torch.nn.Linear(10, 10)
    N = 5
    cloned_modules = clones(module, N)
    
    assert len(cloned_modules) == N, f"Expected {N} cloned modules, got {len(cloned_modules)}"
    for i in range(N):
        assert isinstance(cloned_modules[i], torch.nn.Linear), f"Cloned module {i} is not a Linear layer"
        assert cloned_modules[i] != module, f"Cloned module {i} is the same as the original module"