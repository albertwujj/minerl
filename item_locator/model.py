import torch
from torch import nn
from torchvision.models.resnet import resnet34
import numpy as np


class LocationModel():
    def __init__(self):
        self.resnet = resnet34(pretrained=True)
        self.linear = nn.Linear(1000, 3)
        self.optimiser = torch.optim.Adam(list(self.resnet.parameters()) + list(self.linear.parameters()), lr=.0003, eps=1e-6)

    def predict(self, obses):
        outs = self.linear(self.resnet(obses)) # shape (batch, 3)
        locations, angles = outs[:, :2], outs[:, 2:]
        return locations, angles

    def train(self, obses, locations, angles):
        labels = np.concatenate([locations, angles], axis=-1)
        res_out = self.linear(self.resnet(obses)) # shape (batch, 3)
        loss = ((res_out - labels) ** 2).mean()
        self.resnet.zero_grad()
        self.linear.zero_grad()
        loss.backward()
        self.optimiser.step()

