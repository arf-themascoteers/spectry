import torch
import numpy as np

v = torch.tensor([1, 2, 3, 5, 7])
v1 = v[1:] - v[:-1]

print(v1)