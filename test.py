from layer import Unet_layers
import torch

model = Unet_layers()

testImg = torch.randn(16, 3, 1497, 1497)

out = model(testImg)
print(f'Output shape: {out.shape}')

