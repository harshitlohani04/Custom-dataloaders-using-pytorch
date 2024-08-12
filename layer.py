import torch.nn as nn
from typing import Optional

class Unet_layers(nn.Module):
    def __init__(self, pool_kernel: Optional[int] = 2, pool_padding: Optional[int] = 0,
                 pool_stride: Optional[int] = 1,kernel_size: Optional[int] = 2,
                 stride: Optional[int]=1, padding:Optional[int]=0, base_output:Optional[int]=64) -> None:
        # Defining the params for the Convolutional Layer
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        # Defining the params for the Pooling Layer
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

        # Initializing the model base
        self.base_output = base_output
        self.bottleneck = None
    
    def createModel_downscaling(self, numOfLayers, inpdims):
        out_dims = self.base_output
        layers = []
        for _ in range(numOfLayers):
            layers.append(nn.Conv2d(inpdims, out_dims, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            layers.append(nn.ReLU(inplace = True))
            layers.append(nn.Conv2d(out_dims, out_dims, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            layers.append(nn.ReLU(inplace = True))
            layers.append(nn.MaxPool2d(kernel_size = self.pool_kernel, padding = self.pool_padding, stride = self.pool_stride))
            out_dims *= 2

        self.base_output = out_dims
        upscale_model = nn.Sequential(*layers)
        return upscale_model
    
    def createModel_upscaling(self, numOfLayers):
        outdims = self.base_output
        layers = []
        for _ in range(numOfLayers):
            layers.append(nn.ConvTranspose2d(outdims, outdims/2, kernel_size=self.pool_kernel, stride=self.pool_stride, padding=self.pool_padding))
            layers.append(nn.Conv2d(outdims, outdims/2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            layers.append(nn.ReLU(inplace = True))
            layers.append(nn.Conv2d(outdims/2, outdims/2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            layers.append(nn.ReLU(inplace = True))
            outdims /= 2

        self.base_output = outdims
        downscale_model = nn.Sequential(*layers)
        return downscale_model
    
    def forward(self, x):
        pass



