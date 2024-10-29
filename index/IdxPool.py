# Implementation of index pooling as given in
# "MultiScale Probability Map guided Index Pooling with Attention-based learning for Road and 
# Building Segmentation" paper

import torch.nn as nn
import torch.functional as F
import torch

'''
For exact working of the implemented pooling method refer: https://arxiv.org/abs/2302.09411.

The method implemented has the following params:
- kernel_size --> Defines the kernel size of the pooling layer.
'''

class IdxPool(nn.Module):
    def __init__(self, kernel_size):
        super(IdxPool, self).__init__()
        self.kernel_size = kernel_size
        self.custom_kernel = [torch.tensor([[[[1, 0], [0, 0]]]], dtype = torch.float32),
                              torch.tensor([[[[0, 1], [0, 0]]]], dtype = torch.float32),
                              torch.tensor([[[[0, 0], [1, 0]]]], dtype = torch.float32), 
                              torch.tensor([[[[0, 0], [0, 1]]]], dtype = torch.float32)
                            ]

    def forward(self, x):
        _, channels, _, _ = x.size()
        conv = nn.Conv2d(channels, 4*channels, kernel_size=2, bias=False, stride=2) # bias = False for simplicity
        for i, kernel in enumerate(self.custom_kernel):
            with torch.no_grad():
                conv.weight.data = kernel
            conv.weight.requires_grad = False # So that weights are not updated
            out = conv(x)
            if i == 0:
                tensor = out
            else:
                tensor = torch.cat((tensor, out), dim = 1)
        return tensor
        