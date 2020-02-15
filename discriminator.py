import torch
import torchvision
import torchvision.datasets as ds
import torchvision.transforms as transforms

augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # an upside-down Van Gogh is still a Van Gogh
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])  # normalize to a guassian distribution for all channels.
])

# in this directory, I have colourGogh and openImages
dataset = ds.ImageFolder(".", transform=augmentations)
