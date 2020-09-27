import torch
# import wandb
import network
import torchvision
import torch.nn as nn
import torchvision.datasets as ds
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dummyNetwork import DummyDiscriminator, DummyGenerator
from singleClassDataset import singleClassDataset


conf = {
    "epochs": 60,
    "batch_size": 1,
    "learning_rate": 0.001,
    "momentum": 0.9,
    "dropout": 0.05
}
# wandb.init(project="gogh", config=conf)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # an upside-down Van Gogh is still a Van Gogh
    transforms.RandomVerticalFlip(),  # an upside-down Van Gogh is still a Van Gogh
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to a guassian distribution for all channels.
])

# in this directory, I have colourGogh and openImages
Adataset = singleClassDataset(".", split="trainset",
                              transforms=augmentations,
                              dimensions=(256,256),
                              classname="openImages")
Bdataset = singleClassDataset(".", split="trainset",
                              transforms=augmentations,
                              dimensions=(256,256),
                              classname="colourGogh")

Adataloader = DataLoader(Adataset, batch_size=conf["batch_size"], shuffle=False, num_workers=2)
Bdataloader = DataLoader(Bdataset, batch_size=conf["batch_size"], shuffle=False, num_workers=2)

Dis_B = DummyDiscriminator(3)

Gen_A2B = DummyGenerator(3, 3)
Gen_B2A = DummyGenerator(3, 3)

criterion = nn.MSELoss()
reconstruction_loss = nn.L1Loss()


optimizerDisB= optim.SGD(Dis_B.parameters(), lr=conf["learning_rate"], momentum=0.9)
optimizerGen_A2B= optim.SGD(Gen_A2B.parameters(), lr=conf["learning_rate"], momentum=0.9)
optimizerGen_B2A= optim.SGD(Gen_B2A.parameters(), lr=conf["learning_rate"], momentum=0.9)

for epoch in range(conf["epochs"]):
    l = len(Adataloader)
    for i, Aimages in enumerate(Adataloader):
        trueB = torch.unsqueeze(Bdataset.__getitem__(i), 0) # also get a batch of B images
        print(f"[{i}/{l}], epoch {epoch}.")

        fakeB = Gen_A2B(Aimages)

        mutatedA = Gen_B2A(fakeB)

        A2A_loss = reconstruction_loss(mutatedA, Aimages)

        # cycle is done, now for training the disciminator

        fakeScores = torch.mean(Dis_B(fakeB), 0) # mean per batch for a score from 0 to 1 of "Goghy-ness"
        realScores = torch.mean(Dis_B(trueB), 0)

        discB_fakes = criterion(fakeScores, torch.zeros(conf["batch_size"]))
        discB_reals = criterion(realScores, torch.ones(conf["batch_size"]))

        discLoss = discB_fakes + discB_reals

        discLoss.backward(retain_graph=True) # backprop of the discriminator loss (which also affects the generator)
        A2A_loss.backward() # backprop of the reconcstruction loss (affects generators only)

        optimizerDisB.step()
        optimizerGen_A2B.step()
        optimizerGen_B2A.step()

print("finished training")
