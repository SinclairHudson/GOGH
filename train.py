import torch
import wandb
import network
import torchvision
import torch.nn as nn
import torchvision.datasets as ds
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
import torch.optim as optim
from network import *
from singleClassDataset import *


conf = {
    "epochs": 60,
    "batch_size": 1,
    "learning_rate": 0.001,
    "momentum": 0.9,
    "dropout": 0.05
}
wandb.init(project="gogh", config=conf)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # an upside-down Van Gogh is still a Van Gogh
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to a guassian distribution for all channels.
])

# in this directory, I have colourGogh and openImages
Adataset = singleClassDataset(".", split="trainset", classname="colourGogh")
Bdataset = singleClassDataset(".", split="trainset", classname="openImages")

Adataloader = DataLoader(Adataset, batch_size=conf["batch_size"], shuffle=True, num_workers=2)
Bdataloader = DataLoader(Bdataset, batch_size=conf["batch_size"], shuffle=True, num_workers=2)

Dis_A = Discriminator(3)
Dis_B = Discriminator(3)

Gen_A2B = Generator(3, 3)
Gen_B2A = Generator(3, 3)

criterion = nn.MSELoss()
reconstruction_loss = nn.L1Loss()


optimizer = optim.SGD(net.parameters(), lr=conf["learning_rate"], momentum=0.9)

for epoch in range(conf["epochs"]):
    l = len(Adataloader)
    for i, Aimages in enumerate(Adataloader):
        Bimages = Bdataloader.__getitem__(i)
        print(f"[{i}/{l}], epoch {epoch}.")

        fakeB = Gen_A2B(Aimages)
        fakeA = Gen_B2A(Bimages)

        mutatedA = Gen_B2A(fakeB)
        mutatedB = Gen_A2B(fakeA)

print("finished training")
