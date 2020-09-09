import torch.nn as nn
import torch
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from torch.distributions import transforms

import numpy as np
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import PowerTransform
from torch.distributions import Normal

class cnn(nn.Module):
    def __init__(self,output_dim):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, output_dim)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)

    def forward(self, x):
        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.bn2(F.relu(F.max_pool2d(self.conv2(x), 2)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ffrelu(nn.Module):
    def __init__(self,input_dim, output_dim,H1,H2):
        super(ffrelu, self).__init__()
        self.fc1 = nn.Linear(input_dim, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# for binary classification
class logistic(nn.Module):
    def __init__(self, input_dim, bias=True):
        super(logistic, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, x):
        x = self.linear(x)
        return torch.sigmoid(x)

class tanh(nn.Module):
    def __init__(self, input_dim, output_dim, H):
        super(tanh, self).__init__()
        self.fc1 = nn.Linear(input_dim, H, bias=False)
        self.fc2 = nn.Linear(H, output_dim, bias=False)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class reducedrank(nn.Module):
    def __init__(self, input_dim, output_dim, H):
        super(reducedrank, self).__init__()
        self.fc1 = nn.Linear(input_dim, H, bias=False)
        self.fc2 = nn.Linear(H, output_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def pyro_tanh(X, D_H, beta, Bmap=None, Amap=None):

    D_X = X.shape[1]

    # sample first layer (we put unit normal priors on all weights)
    if Amap is not None:
        A = pyro.sample("A", dist.Normal(Amap, torch.ones((D_X, D_H))))  # D_X D_H
    else:
        A = pyro.sample("A", dist.Normal(torch.zeros((D_X, D_H)), torch.ones((D_X, D_H))))  # D_X D_H
    z1 = torch.tanh(torch.matmul(X, A))   # N D_H  <= first layer of activations

    # sample second layer
    if Bmap is not None:
        B = pyro.sample("B", dist.Normal(Bmap, torch.ones((D_H, 1))))  # D_H D_H
    else:
        B = pyro.sample("B", dist.Normal(torch.zeros((D_H, 1)), torch.ones((D_H, 1))))  # D_H D_H
    z2 = torch.matmul(z1, B)

    # TODO: transform not working
    # return pyro.sample("Y", dist.TransformedDistribution(dist.Normal(z2, 1.0), transforms.PowerTransform(beta)))
    return pyro.sample("Y", dist.Normal(z2, 1/np.sqrt(beta)))


def conditioned_pyro_tanh(pyro_tanh, X, Y, D_H, beta, Bmap, Amap):
    
    return poutine.condition(pyro_tanh, data={"Y": Y})(X, D_H, beta, Bmap, Amap)


def pyro_rr(X, Y, D_H, beta, Bmap=None, Amap=None):

    D_X = X.shape[1]
    D_Y = Y.shape[1]

    # sample first layer (we put unit normal priors on all weights)
    if Amap is not None:
        A = pyro.sample("A", dist.Normal(Amap, torch.ones((D_X, D_H))))  # D_X D_H
    else:
        A = pyro.sample("A", dist.Normal(torch.zeros((D_X, D_H)), torch.ones((D_X, D_H))))  # D_X D_H
    z1 = torch.matmul(X, A)   

    # sample second layer
    if Bmap is not None:
        B = pyro.sample("B", dist.Normal(Bmap, torch.ones((D_H, D_Y))))  # D_H D_Y
    else:
        B = pyro.sample("B", dist.Normal(torch.zeros((D_H, D_Y)), torch.ones((D_H, D_Y))))  # D_H D_Y
    z2 = torch.matmul(z1, B)

    # TODO: transform not working
    # return pyro.sample("Y", dist.TransformedDistribution(dist.Normal(z2, 1.0), transforms.PowerTransform(beta)))
    return pyro.sample("Y", dist.Normal(z2, 1/np.sqrt(beta)))


def conditioned_pyro_rr(pyro_rr, X, Y, D_H, beta, Bmap, Amap):
    
    return poutine.condition(pyro_rr, data={"Y": Y})(X, Y, D_H, beta, Bmap, Amap)
