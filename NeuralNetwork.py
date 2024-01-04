import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MLPBiomass(nn.Module):
    def __init__(self, n_input, n_output):
        super(MLPBiomass, self).__init__()
        # hidden 1
        self.fc1 = nn.Linear(in_features=n_input, out_features=256)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)

        # hidden 2
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(128)

        # hidden 3
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(64)

        # linear output
        self.linear_output = nn.Linear(in_features=64, out_features=n_output)

    def forward(self, x):
        # hidden 1
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        # hidden 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        # hidden 3
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.bn3(x)

        # linear output
        output = self.linear_output(x)

        return output


class BiomassDataset(Dataset):
    def __init__(self, data):
        self.data = data

        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
