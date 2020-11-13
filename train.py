import torch
import time
import wandb
import network
import torchvision
import torch.nn as nn
import torchvision.datasets as ds
import torchvision.transforms as transforms
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dummyNetwork import DummyDiscriminator, DummyGenerator
from singleClassDataset import singleClassDataset

torch.autograd.set_detect_anomaly(True)

def tensorsToImages(tensor):
    """
    Returns a numpy array in [0,1], with floats, like a PIL image
    """
    tensor = torch.clamp(tensor, -1.0, 1.0).detach()
    unnormalize = transforms.Normalize((-1, -1, -1), (2, 2, 2)) # [-1, 1] => [0,1]
    images = [np.swapaxes(unnormalize(t).numpy(), 0, 2) for t in tensor]
    return images

end = 0


conf = {
    "epochs": 60,
    "batch_size": 32,
    "disc_learning_rate": 0.01,
    "gen_learning_rate": 0.001,
    "momentum": 0.9,
    "dropout": 0.05,
    "image_dim": 512,
    "reconstruction_loss_factor": 10,
    "adversarial_loss_factor": 1,
}

wandb.init(project="gogh", config=conf)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # an upside-down Van Gogh is still a Van Gogh
    transforms.RandomVerticalFlip(),  # an upside-down Van Gogh is still a Van Gogh
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to a guassian distribution for all channels.
])

# in this directory, I have colourGogh and openImages
d = conf["image_dim"]
Adataset = singleClassDataset(".", split="trainset",
                              transforms=augmentations,
                              dimensions=(d,d),
                              classname="openImages")
Bdataset = singleClassDataset(".", split="trainset",
                              transforms=augmentations,
                              dimensions=(d,d),
                              classname="colourGogh")

Adataloader = DataLoader(Adataset, batch_size=conf["batch_size"], shuffle=False, num_workers=3)
Bdataloader = DataLoader(Bdataset, batch_size=conf["batch_size"], shuffle=False, num_workers=3)

Dis_B = DummyDiscriminator(3).to(device)

Gen_A2B = DummyGenerator(3, 3).to(device)
Gen_B2A = DummyGenerator(3, 3).to(device)

criterion = nn.MSELoss()
reconstruction_loss = nn.L1Loss()


optimizerDisB= optim.SGD(Dis_B.parameters(), lr=conf["disc_learning_rate"], momentum=0.9)
optimizerGen_A2B= optim.SGD(Gen_A2B.parameters(), lr=conf["gen_learning_rate"], momentum=0.9)
optimizerGen_B2A= optim.SGD(Gen_B2A.parameters(), lr=conf["gen_learning_rate"], momentum=0.9)

for epoch in range(conf["epochs"]):
    l = len(Adataloader)
    for i, Aimages in enumerate(Adataloader):
        trueB = next(iter(Bdataloader))

        start = time.time()
        trueB = trueB.to(device)
        Aimages = Aimages.to(device)

        fakeB = Gen_A2B(Aimages)  # generating the examples
        mutatedA = Gen_B2A(fakeB)


        # mean per image for a score from 0 to 1 of "Goghy-ness"
        fakeScores = torch.mean(Dis_B(fakeB), (2,3), keepdim=True).squeeze()
        realScores = torch.mean(Dis_B(trueB), (2,3), keepdim=True).squeeze()

        discB_fakes = criterion(fakeScores, torch.zeros(conf["batch_size"]).to(device))
        discB_reals = criterion(realScores, torch.ones(conf["batch_size"]).to(device))

        discLoss = discB_fakes + discB_reals
        adverse_loss = criterion(fakeScores, torch.ones(conf["batch_size"]).to(device))
        A2A_loss = reconstruction_loss(mutatedA, Aimages)
        genLoss = conf["reconstruction_loss_factor"]* A2A_loss + \
                conf["adversarial_loss_factor"] * adverse_loss



        # remove the gradients that discLoss backprop put into A2B generator
        # the discriminator messing up should not negatively affect Gen_A2B!

        discLoss.backward(retain_graph=True) # backprop of the discriminator loss
        Gen_A2B.zero_grad()
        Gen_B2A.zero_grad()
        for param in Dis_B.parameters():
            param.requires_grad = False
        genLoss.backward()
        for param in Dis_B.parameters():
            param.requires_grad = True


        optimizerGen_A2B.step()
        optimizerGen_B2A.step()
        optimizerDisB.step()

        # unnormalize
        fakeB = torch.clamp(fakeB, -1.0, 1.0)
        echo = start - end
        end = time.time()
        delta = end - start
        print(f"[{i}/{l}], epoch {epoch}. D_Loss: {discLoss.item()}, Re_Loss: {A2A_loss.item()}, CompTime: {delta:.2f}, DataTime: {echo:.2f}")

        wandb.log({
            "Discriminator_Loss": discLoss.item(),
            "Reconstruction_Loss": A2A_loss.item(),
            "Adversarial_Loss": adverse_loss.item(),
        })

    wandb.log({
        "Fake B image": [wandb.Image(single) for single in tensorsToImages(fakeB[0:5].cpu())],
        "Real A image": [wandb.Image(single) for single in tensorsToImages(Aimages[0:5].cpu())],
        "Reconstructed A image": [wandb.Image(single) for single in tensorsToImages(mutatedA[0:5].cpu())],
    })



print("finished training")

