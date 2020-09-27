"""
This file defines Dummy models that are easy to run. They're one conv each.
They're used for debugging; very little can go wrong, and they're fast.
They're also non memory-intensive.
"""
import torch.nn as nn
import torch.nn.functional as F



class DummyGenerator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(DummyGenerator, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, output_nc, 3, padding=1)


    def forward(self, x):
        return F.tanh(self.conv1(x))

class DummyDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super(DummyDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, 1, 3)


    def forward(self, x):
        x = self.conv1(x)
        return F.sigmoid(x)


