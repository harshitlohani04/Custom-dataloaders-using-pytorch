import os
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import random

train_images = "./png/train"
train_mask_images = "./png/train_labels"


def mapping(csvPath):
    '''
    0 --> Background
    1 --> Building

    '''
    data = pd.read_csv(csvPath)
    bwmapping = {}
    index = 0
    for row in data.to_numpy():
        bwmapping[index] = (row[1], row[2], row[3])
        index+=1

    return bwmapping


class CustomDataloader(Dataset):
    def __init__(self, image_dir, mask_dir, augment = True, transforms = None) -> None:
        super().__init__()
        self.image_dir= image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.transforms = transforms
        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))

        self.mapping_rgb = mapping("label_class_dict.csv")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        ''' 
        Load a particular image from the main dataset
        at the given index of the method.
        Do the same with the mask images.
        '''
        imgPath = os.path.join(self.image_dir, self.images[index])
        maskPath = os.path.join(self.mask_dir, self.masks[index])

        img = Image.open(imgPath).convert("RGB")
        mask = Image.open(maskPath).convert("L")

        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)

        if self.augment:
            if random.random()>0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

        # Normalizing the image if needed
        if self.transforms:
            img = self.transforms(img)
            
        mask = (mask>0).float()

        return img, mask
