import torch
import os
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

train_images = "./png/train"
train_mask_images = "./png/train_labels"


class CustomDataloader(Dataset):
    def __init__(self, image_dir, mask_dir, augment = True, transforms = None) -> None:
        super().__init__()
        self.image_dir= image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.transforms = transforms
        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))

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

        if self.augment:
            return True
        else:
            
        return super().__getitem__(index)
