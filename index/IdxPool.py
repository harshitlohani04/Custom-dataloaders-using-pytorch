import torch.nn as nn
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
        self.custom_kernel = [
            torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float32),
            torch.tensor([[[[0, 1], [0, 0]]]], dtype=torch.float32),
            torch.tensor([[[[0, 0], [1, 0]]]], dtype=torch.float32),
            torch.tensor([[[[0, 0], [0, 1]]]], dtype=torch.float32)
        ]

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        output = []

        # Iterate over each custom kernel
        for kernel in self.custom_kernel:
            kernel = kernel.expand(channels, 1, *kernel.shape[2:])  # Shape becomes [channels, 1, 2, 2]
            conv = nn.Conv2d(
                in_channels=channels, out_channels=channels, kernel_size=self.kernel_size, stride=2,
                padding=0, groups=channels, bias=False
            )
            
            with torch.no_grad():
                conv.weight.data = kernel
            conv.weight.requires_grad = False  # Ensure these weights are fixed

            out = conv(x)
            output.append(out)
        
        tensor = torch.cat(output, dim=1) # Concatenate all outputs along the channel dimension

        return tensor
