import numpy as np
import torch.utils
import torch.nn as nn
import torch.utils.data
from layer import Unet_layers
from final_layer import UNet
from customdataloader import CustomDataloader
import torch
from torch.utils.data import Subset
from model import UnetGenerator
from torch.autograd import Variable


if __name__ == "__main__":
    train_images = "./png/train"
    train_mask_images = "./png/train_labels"

    img_data = CustomDataloader(image_dir=train_images, mask_dir=train_mask_images)
    imgBatch = torch.utils.data.DataLoader(img_data, batch_size=16, shuffle=False, num_workers=4)
    generator = UNet()

    epochs = 5
    lr = 0.001

    error = nn.CrossEntropyLoss()
    if not generator.parameters():
        raise ValueError("Was Expecting params found nothing")
    else:
        optim = torch.optim.Adam(params=generator.parameters(), lr=lr)
        print("params exist")
        for i in range(epochs):
            for img, masks in imgBatch:
                print("inside the 2nd for loop")
                try:
                    optim.zero_grad()
                    x = img
                    print(x.shape)
                    y_ = masks
                    y_ = y_.squeeze(1).long()
                    print(y_)
                    print(torch.sum(y_))
                    y = generator(x)
                    print(y)
                    print("hi")
                    loss = error(y, y_)
                    print("hello")
                    print(f"Epoch number - {i} ||| Current loss - {loss}")
                    loss.backward()
                    optim.step()

                except Exception as e:
                    print(f"An Unexpected error occurred. Error --> {e}")
