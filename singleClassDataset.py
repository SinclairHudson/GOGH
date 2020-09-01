import torchvision
import torch
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

class singleClassDataset(Dataset):
    """
    This dataset only loads images from a single domain.
    Initialize two, one from each domain, for unpaired image translation

    """
    def __init__(self, path, split="trainset", classname="colourGogh", dimensions=(256,256)):
        self.path = path
        self.split = split
        self.dimensions = dimensions
        self.classname = classname
        self.images = list(os.listdir(f"{path}/{split}/{classname}"))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize distribution for all channels.
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        assert 0 <= index <= len(self.images) # make sure we're in bounds
        image = Image.open(f"{self.path}/{self.split}/{self.classname}/{self.images[index]}")
        image = image.resize(dimensions)

        # height, then width
        in_tensor = self.transforms(image)
        return in_tensor
