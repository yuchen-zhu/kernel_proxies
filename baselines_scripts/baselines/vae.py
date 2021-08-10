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
    def __init__(self, hidden_1, dim_y, dim_u, dim_a, dim_w, dim_z, C_init, epochs, batch_size, lr):
        super(VAE, self).__init__()

        self.C = torch.as_tensor(C_init).float()
        self.C.requires_grad = True
        in_features = dim_w + dim_a + dim_z + dim_y

        self.fc1 = nn.Linear(in_features, hidden_1)
        self.fc21 = nn.Linear(hidden_1, dim_u)
        self.fc22 = nn.Linear(hidden_1, dim_u)
        self.fc_z = nn.Linear(dim_u, dim_z)
        self.fc_w = nn.Linear(dim_u, dim_w)
        self.fc_a = nn.Linear(dim_u + dim_z, dim_a)
        self.fc_y1 = nn.Linear(dim_u + dim_a + dim_w, dim_y)
        self.fc_y2 = nn.Linear(dim_y, dim_y)

        self.optimizer = optim.Adam(list(self.parameters()), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs

    def encode(self, w, a, z, y):

        all = torch.cat([w, a, z, y], dim=-1)

        h1 = F.relu(self.fc1(all))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(self.C*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    # def decode_x(self, u):
    #     x = self.fc_x(u)
    #     return x

    def decode_z(self, u):
        # ux = torch.cat([u, x], dim=-1)
        z = self.fc_z(u)
        return z

    def decode_w(self, u):
        # ux = torch.cat([u, x], dim=-1)
        w = self.fc_w(u)
        return w

    def decode_a(self, u, z):
        uz = torch.cat([u, z], dim=-1)
        # pdb.set_trace()
        a = self.fc_a(uz)
        a = torch.sigmoid(a)
        a = torch.bernoulli(a)
        return a

    def decode_y(self, u, a, w):
        uaw = torch.cat([u, a, w], dim=-1)
        # pdb.set_trace()
        h_y = F.relu(self.fc_y1(uaw))

        return self.fc_y2(h_y)

    def forward(self, w, a, z, y):
        mu, logvar = self.encode(w, a, z, y)
        u = self.reparameterize(mu, logvar)
        z_ = self.decode_z(u)
        w_ = self.decode_w(u)
        a_ = self.decode_a(u, z_)
        y_ = self.decode_y(u, a_, w_)
        wazy = torch.cat([w_, a_, z_, y_], dim=-1)

        return wazy, mu, logvar

    # def _fit(self):
    #     'The training loop.'
    #     pass

    def fit(self, a, y, z, w):
        self._model.train()
        n_data = a.shape[0]
        permutation = torch.randperm(n_data)
        train_loss = 0
        for epoch in range(self.epochs):
            for i in range(0, n_data, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_a, batch_y, batch_z, batch_w = a[indices], y[indices], z[indices], w[indices]
                w, a, z, y = batch_w.to(self._device), batch_a.to(self._device), batch_z.to(self._device), batch_y.to(self._device)
                x = torch.cat([w, a, z, y], dim=1)

                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self._model(w, a, z, y)
                loss = self._loss_function(recon_batch, x, mu, logvar, self._model.C)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

                if i % self._log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, self.batch_idx * len(x), n_data,
                               100. * self.batch_idx / (n_data // self.batch_size),
                               loss.item() / len(x)))

            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / n_data))

        return train_loss