import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class CustomDataloader(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
