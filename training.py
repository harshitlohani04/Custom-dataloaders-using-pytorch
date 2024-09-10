import torch.utils
import torch.nn as nn
import torch.utils.data
from layer import Unet_layers
from final_layer import UNet
from customdataloader import CustomDataloader
import torch
from torch.utils.data import Subset

train_images = "./png/train"
train_mask_images = "./png/train_labels"

#Incorporating few shot learning by sampling the data


img_data = CustomDataloader(image_dir=train_images, mask_dir=train_mask_images)
imgBatch = torch.utils.data.DataLoader(img_data, batch_size=16, shuffle=False, num_workers=4)


if __name__ == "__main__":
    layer_instance = UNet()
    for img, masks in imgBatch:
        print(img.shape)
        print(masks.shape)

    epochs = 1
    lr = 0.01

    error = nn.CrossEntropyLoss()
    if not layer_instance.parameters():
        raise ValueError("Was Expecting params found nothing")
    else:
        optim = torch.optim.Adam(params=layer_instance.parameters(), lr=lr)

        for i in range(epochs):
            for img, masks in imgBatch:
                print(img.shape)
                print(masks.shape)
                outputs = layer_instance(img)
                loss = error(outputs, masks)
                loss.backward()
                optim.step()
