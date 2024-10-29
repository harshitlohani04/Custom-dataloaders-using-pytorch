# Implementation of index pooling as given in
# "MultiScale Probability Map guided Index Pooling with Attention-based learning for Road and 
# Building Segmentation" paper

import torch.nn as nn
import torch.functional as F
import torch

'''
For exact working of the implemented pooling method refer: https://arxiv.org/abs/2302.09411.

The method implemented has the following params:
kernel_size --> Defines the kernel size of the pooling layer.
'''

class IdxPool(nn.Module):
    def __init__(self, kernel_size):
        super(IdxPool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):

        pass