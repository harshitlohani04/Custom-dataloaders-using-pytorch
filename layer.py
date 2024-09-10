import torch
import torch.nn as nn
from typing import Optional

class Unet_layers(nn.Module):
    def __init__(self, pool_kernel: Optional[int] = 2, pool_padding: Optional[int] = 0,
                 pool_stride: Optional[int] = 1,kernel_size: Optional[int] = 2,
                 stride: Optional[int]=1, padding:Optional[int]=0, base_output:Optional[int]=64,
                 numOflayers = 4) -> None:
        
        super(Unet_layers, self).__init__()
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
        self.numOflayers = numOflayers

        # Initialize lists to store layers
        self.downscaling_layers = nn.ModuleList()
        self.bottleneck_layers = nn.ModuleList()
        self.upscaling_layers = nn.ModuleList()

        # Create layers
        self.createModel_downscaling()
        self.bottleneck()
        self.createModel_upscaling()

    def createModel_downscaling(self):
        out_dims = self.base_output
        layers = []
        for _ in range(self.numOflayers):
            layers.append(nn.Conv2d(3, out_dims, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            layers.append(nn.ReLU(inplace = True))
            layers.append(nn.Conv2d(out_dims, out_dims, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            layers.append(nn.ReLU(inplace = True))
            layers.append(nn.MaxPool2d(kernel_size = self.pool_kernel, padding = self.pool_padding, stride = self.pool_stride))

            inpdims = out_dims
            out_dims *= 2

        self.base_output = out_dims
        down_model = nn.Sequential(*layers)
        self.downscaling_layers.append(down_model)
        return down_model


    def bottleneck(self):
        bottleneck = nn.Sequential(
            nn.Conv2d(self.base_output, self.base_output*2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_output*2, self.base_output*2, kernel_size=self.kernel_size, stride = self.stride, padding=self.padding),
            nn.ReLU(inplace=True)
        )
        self.bottleneck_layers.append(bottleneck)
        self.base_output*=2
        return bottleneck


    def createModel_upscaling(self):
        outdims = self.base_output
        numOfLayers = self.numOflayers
        layers = []
        for i in range(numOfLayers):
            if i!=numOfLayers-1:
                layers.append(nn.ConvTranspose2d(outdims, outdims//2, kernel_size=self.pool_kernel, stride=self.pool_stride, padding=self.pool_padding))
                layers.append(nn.Conv2d(outdims, outdims//2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
                layers.append(nn.ReLU(inplace = True))
                layers.append(nn.Conv2d(outdims//2, outdims//2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
                layers.append(nn.ReLU(inplace = True))
                outdims //= 2
            else:
                layers.append(nn.ConvTranspose2d(outdims, outdims//2, kernel_size=self.pool_kernel, stride=self.pool_stride, padding=self.pool_padding))
                layers.append(nn.Conv2d(outdims, outdims//2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
                layers.append(nn.ReLU(inplace = True))
                layers.append(nn.Conv2d(outdims//2, outdims//2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
                layers.append(nn.ReLU(inplace = True))
                layers.append(nn.Conv2d(outdims//2, 1, kernel_size=1, stride=1))
                outdims = 1

        self.base_output = outdims
        up_model = nn.Sequential(*layers)
        self.upscaling_layers.append(up_model)
        return up_model

    def forward(self, x):
        numOflayers = self.numOflayers
        encLayers = self.createModel_downscaling()
        prev = 0
        skip = []
        prevResult = x
        for i in range(5, numOflayers*5+1, 5):
            enc = encLayers[prev:i](prevResult)
            skip.append(enc)
            prev = i
            prevResult = enc
        
        bottleneck = self.bottleneck()
        encPathFinal = bottleneck(prevResult)
        decPath = self.createModel_upscaling()
        out = encPathFinal
        for i in range(numOflayers):
            if i!=numOflayers-1:
                dec = decPath[i*5](out)
                dec = torch.cat((dec, skip[-(i+1)]), dim = 1)
                dec = decPath[i*5+1:i*5+5](dec)
            else:
                dec = decPath[i*5](out)
                dec = torch.cat((dec, skip[-(i+1)]), dim = 1)
                dec = decPath[i*5+1:i*5+6](dec)
            out = dec

        return torch.sigmoid(out)

model = Unet_layers(pool_stride=2, pool_kernel=2, pool_padding=1, padding=0, kernel_size=3, stride=1)
for name, param in model.named_parameters():
    print(name, param.shape)
