from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from scipy.integrate import nquad
import mcint
from src.utils.utils import rejection_sampling
from datetime import datetime
import sys

import pdb

class VAEDataset(Dataset):
    """Dataset of (Z, A, Z_1, Y) for the VAE."""

    def __init__(self, Z_np, A_np, W_np, Y_np, X_np, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # self.data = pd.concat([self.Ws_train, self.As_train, self.Zs_train, self.Ys_train], axis=1, keys=['Ws', 'As', 'Zs', 'Ys'])

        self.transform = transform
        datasize = Z_np.shape[0]

        self.W = self.transform(torch.as_tensor(W_np).float(), key='W')
        self.A = self.transform(torch.as_tensor(A_np).float(), key='A')
        self.Z = self.transform(torch.as_tensor(Z_np).float(), key='Z')
        self.Y = self.transform(torch.as_tensor(Y_np).float(), key='Y')
        if X_np is None:
            self.X = torch.as_tensor(np.tile([[]], [datasize, 1])).float()
        else:
            self.X = self.transform(torch.as_tensor(X_np).float(), key='X')
        # print('w size: ', self.W.size())
        # assert self.W.size()[0] == 8000 and len(self.W.size()) == 2
        # assert self.A.size()[0] == 8000 and len(self.A.size()) == 2
        # assert self.Z.size()[0] == 8000 and len(self.Z.size()) == 2
        # assert self.Y.size()[0] == 8000 and len(self.Y.size()) == 2


    def __len__(self):
        # assert len(self.W) == 8000
        return len(self.W)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        samples = (self.W[idx],
                   self.A[idx],
                   self.Z[idx],
                   self.Y[idx],
                   self.X[idx])

        return samples


class VAE(nn.Module):
    def __init__(self, in_features, hidden_1, dim_y, dim_u, dim_a, dim_w, dim_z, dim_x, C=0.5):
        super(VAE, self).__init__()

        self.C = torch.as_tensor(C).float()
        self.C.requires_grad = True

        self.fc1 = nn.Linear(in_features, hidden_1)
        self.fc21 = nn.Linear(hidden_1, dim_u)
        self.fc22 = nn.Linear(hidden_1, dim_u)
        self.fc_z = nn.Linear(dim_u + dim_x, dim_z)
        self.fc_w = nn.Linear(dim_u + dim_x, dim_w)
        self.fc_a = nn.Linear(dim_u + dim_z + dim_x, dim_a)
        self.fc_y1 = nn.Linear(dim_u + dim_a, dim_y)
        self.fc_y2 = nn.Linear(dim_y, dim_y)
        self.fc_x = nn.Linear(dim_u, dim_x)

    def encode(self, w, a, z, y, x):

        all = torch.cat([w, a, z, y, x], dim=-1)

        h1 = F.relu(self.fc1(all))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(self.C*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode_x(self, u):
        x = self.fc_x(u)
        return x

    def decode_z(self, u, x):
        ux = torch.cat([u, x], dim=-1)
        z = self.fc_z(ux)
        return z

    def decode_w(self, u, x):
        ux = torch.cat([u, x], dim=-1)
        w = self.fc_w(ux)
        return w

    def decode_a(self, u, z, x):
        uzx = torch.cat([u, z, x], dim=-1)
        # pdb.set_trace()
        a = self.fc_a(uzx)
        a = torch.sigmoid(a)
        a = torch.bernoulli(a)
        return a

    def decode_y(self, u, a, x):
        uax = torch.cat([u, a, x], dim=-1)
        # pdb.set_trace()
        h_y = F.relu(self.fc_y1(uax))

        return self.fc_y2(h_y)

    def forward(self, w, a, z, y, x):
        mu, logvar = self.encode(w, a, z, y, x)
        u = self.reparameterize(mu, logvar)
        x_ = self.decode_x(u)
        z_ = self.decode_z(u, x_)
        w_ = self.decode_w(u, x_)
        a_ = self.decode_a(u, z_, x_)
        y_ = self.decode_y(u, a_, x_)
        wazyx = torch.cat([w_, a_, z_, y_, x_], dim=-1)

        return wazyx, mu, logvar


