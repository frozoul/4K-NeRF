import torch
import torch.nn as nn


class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=False):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))
