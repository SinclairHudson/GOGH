import torch.nn as nn
import torch
import torchvision
from torch.nn import functional as F
from denseblock import DenseBlock

class Discriminator(nn.Module):
    def __init__(self, dropout=0.00):
        super(Discriminator, self).__init__()
        self.dense0 = DenseBlock(3, 4, 16, drop_prob=dropout)
        self.middle_conv = nn.Conv2d(3 + 4 * 16, 64, 5)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dense1 = DenseBlock(64, 4, 16, drop_prob=dropout)
        self.final_conv = nn.Conv2d(64 + 4 * 16, 1, 5, stride=2)

    def forward(self, x):
        x = self.dense0(x)
        x = self.middle_conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.final_conv(x)
        return x



