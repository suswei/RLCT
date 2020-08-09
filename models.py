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


# feedforward relu network with D_H hidden units, at "temperature" 1/beta
def pyro_tanh(X, D_H, beta):

    D_X = X.shape[1]

    # sample first layer (we put unit normal priors on all weights)
    w = pyro.sample("w", dist.Normal(torch.zeros((D_X, D_H)), torch.ones((D_X, D_H))))  # D_X D_H
    z1 = torch.tanh(torch.matmul(X, w))   # N D_H  <= first layer of activations

    # sample second layer
    q = pyro.sample("q", dist.Normal(torch.zeros((D_H, 1)), torch.ones((D_H, 1))))  # D_H D_H
    z2 = torch.matmul(z1, q)

    # TODO: transform not working
    # return pyro.sample("Y", dist.TransformedDistribution(dist.Normal(z2, 1.0), transforms.PowerTransform(beta)))
    return pyro.sample("Y", dist.Normal(z2, 1/np.sqrt(beta)))


def conditioned_pyro_tanh(pyro_tanh, X,Y,D_H,beta):
    return poutine.condition(pyro_tanh, data={"Y": Y})(X, D_H, beta)


def pyro_ffrelu(X, Y, D_H, beta):

    D_X, D_Y = X.shape[1], Y.shape[1]
    prior_std = 10.0

    w1 = pyro.sample("w1", dist.Normal(torch.zeros((D_X, D_H)), prior_std*torch.ones((D_X, D_H))))  # D_X D_H
    b1 = pyro.sample("b1", dist.Normal(torch.zeros((1, D_H)), prior_std*torch.ones((1, D_H))))  # D_X D_H
    z1 = torch.relu(torch.matmul(X, w1)+b1)   # N D_H  <= first layer of activations

    w2 = pyro.sample("w2", dist.Normal(torch.zeros((D_H, D_H)), prior_std*torch.ones((D_H, D_H))))  # D_X D_H
    b2 = pyro.sample("b2", dist.Normal(torch.zeros((1, D_H)), prior_std*torch.ones((1, D_H))))  # D_X D_H
    z2 = torch.relu(torch.matmul(z1, w2)+b2)   # N D_H  <= first layer of activations

    w3 = pyro.sample("w3", dist.Normal(torch.zeros((D_H, D_H)), prior_std*torch.ones((D_H, D_H))))  # D_X D_H
    b3 = pyro.sample("b3", dist.Normal(torch.zeros((1, D_H)), prior_std*torch.ones((1, D_H))))  # D_X D_H
    z3 = torch.relu(torch.matmul(z2, w3)+b3)   # N D_H  <= first layer of activations

    w4 = pyro.sample("w4", dist.Normal(torch.zeros((D_H, D_H)), prior_std*torch.ones((D_H, D_H))))  # D_X D_H
    b4 = pyro.sample("b4", dist.Normal(torch.zeros((1, D_H)), prior_std*torch.ones((1, D_H))))  # D_X D_H
    z4 = torch.relu(torch.matmul(z3, w4)+b4)   # N D_H  <= first layer of activations

    w5 = pyro.sample("w5", dist.Normal(torch.zeros((D_H, D_Y)), prior_std*torch.ones((D_H, D_Y))))  # D_X D_H
    b5 = pyro.sample("b5", dist.Normal(torch.zeros((1, D_Y)), prior_std*torch.ones((1, D_Y))))  # D_X D_H
    z5 = torch.matmul(z4, w5)+b5  # N D_H  <= first layer of activations

    pyro.sample("Y", dist.Normal(z5, 1 / np.sqrt(beta)), obs=Y)