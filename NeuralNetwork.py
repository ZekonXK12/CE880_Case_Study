import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MLPBiomass(nn.Module):
    def __init__(self, n_input=8, n_output=7):
        super(MLPBiomass,self).__init__()
        self.fc1 = nn.Linear(in_features=n_input, out_features=128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.relu3=nn.ReLU()
        self.linear=nn.Linear(in_features=32, out_features=n_output)

    def forward(self, x):
        # hidden 1
        output = self.fc1(x)
        output = self.relu1(output)
        # # hidden 2
        output = self.fc2(output)
        output = self.relu2(output)
        # hidden 3
        output = self.fc3(output)
        output=self.relu3(output)
        # linear output
        output=self.linear(output)

        return output


class BiomassDataset(Dataset):
    def __init__(self, data):
        self.data = data

        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
