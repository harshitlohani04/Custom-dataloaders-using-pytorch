import torch.nn as nn
from typing import Optional

class Unet_layers(nn.Module):
    def __init__(self, inp_dims, kernel_size: Optional[int] = 2, stride: Optional[int]=1, padding:Optional[int]=0) -> None:
        self.inp_dims = inp_dims
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def layers(self):
        pass
