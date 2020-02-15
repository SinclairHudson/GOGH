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


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


conf = {
    "epochs": 60,
    "batch_size": 1,
    "learning_rate": 0.008,
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
trainingset = ds.ImageFolder("trainset", transform=augmentations)

trainloader = torch.utils.data.DataLoader(trainingset, batch_size=conf["batch_size"], shuffle=True, num_workers=3,
                                          collate_fn=my_collate)

validationset = ds.ImageFolder("testset", transform=augmentations)

validloader = torch.utils.data.DataLoader(trainingset, batch_size=conf["batch_size"], shuffle=True, num_workers=3,
                                          collate_fn=my_collate)

net = network.Discriminator(dropout=conf["dropout"])
wandb.watch(net)
net.to(device)

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


net.final_conv.register_forward_hook(get_activation('final_conv'))

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=conf["learning_rate"], momentum=0.9)

for epoch in range(conf["epochs"]):
    # test before every epoch
    print("Starting epoch " + str(epoch))
    net.eval()
    running_loss = 0
    for i, data in enumerate(validloader, 0):
        inputs, labels = data
        print(labels)
        realLabels = labels
        print(type(inputs))
        output = net(inputs[0].unsqueeze(0).to(device))
        scores = torch.mean(output, dim=0)  # mean value for each of the images in the batch
        yhat = F.sigmoid(scores)  # set scores from 0 to 1
        loss = criterion(scores, yhat)
        running_loss += loss.item()

    wandb.log({"test_loss": running_loss})

    net.train()  # back to training mode

    for i, data in enumerate(trainloader, 0):
        # rotation = random.choice([0, 90, 180, 270])
        # rotate tensor
        inputs, labels = data
        print(labels)
        print(inputs)
        realLabels = labels
        inputs = torch.FloatTensor(inputs)
        output = net(inputs)
        scores = torch.mean(output, dim=0)  # mean value for each of the images in the batch
        yhat = F.sigmoid(scores)  # set scores from 0 to 1
        loss = criterion(yhat, realLabels)
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            wandb.log({"training_loss": loss})

print("finished training")
