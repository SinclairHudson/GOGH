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
    def __init__(self, path, transforms=None, split="trainset", classname="colourGogh", dimensions=(256,256)):
        self.path = path
        self.split = split
        self.dimensions = dimensions
        self.transforms = transforms
        self.classname = classname
        self.images = list(os.listdir(f"{path}/{split}/{classname}"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        assert 0 <= index <= len(self.images) # make sure we're in bounds
        path = f"{self.path}/{self.split}/{self.classname}/{self.images[index]}"
        image = Image.open(path).convert(mode="RGB") # some of openimages are not RGB
        image = image.resize(self.dimensions)
        if self.transforms:
            image= self.transforms(image)
        # height, then width
        return image
