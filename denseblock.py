import torch
from torch import nn
import torch.nn.functional as F


class CompositeFunction(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0):
        super(CompositeFunction, self).__init__()
        self.drop = nn.Dropout2d(drop_prob)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.BN = nn.BatchNorm2d(out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # The paper differs in this order. It does BN Activation Conv
        # I think it makes more sense this way... Conv BN Activation
        return self.drop(F.relu(self.BN(self.conv(x))))


class DenseBlock(nn.Module):
    def __init__(self, in_channels, repetition, k, drop_prob=0):
        super(DenseBlock, self).__init__()
        self.functions = nn.ModuleList()
        for i in range(repetition):
            layer = CompositeFunction(in_channels + k * i, k, drop_prob)
            self.functions.append(layer)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        for index, function in enumerate(self.functions):
            new = self.functions[index](x)
            x = torch.cat((x, new), dim=1)

        return x


