# %%

import argparse
import numpy as np
import math
import copy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, SubsetRandomSampler
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import Variable

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from matplotlib import pyplot as plt


# %%

class args():

    def __init__(self):
        self.n = 500
        self.batchsize = 500
        self.lr = 0.001
        self.epochs = 500

        self.output_dim = 5

        self.y_std = 1.0
        self.prior_std = 1.0

        self.betasbegin = 0.9
        self.betasend = 1.1
        self.numbetas = 10

        self.R = 25


args = args()

# %%
args.H = args.output_dim
args.input_dim = args.output_dim + 3
args.a_params = torch.transpose(
    torch.cat((torch.eye(args.H), torch.ones([args.H, args.input_dim - args.H], dtype=torch.float32)), 1), 0,
    1)  # input_dim * H
args.b_params = torch.eye(args.output_dim)

m = MultivariateNormal(torch.zeros(args.input_dim), torch.eye(
    args.input_dim))  # the input_dim=output_dim + 3, output_dim = H (the number of hidden units)
X = m.sample(torch.Size([2 * args.n]))
mean = torch.matmul(torch.matmul(X, args.a_params), args.b_params)
y_rv = MultivariateNormal(torch.zeros(args.output_dim), torch.eye(args.output_dim))
y = mean + y_rv.sample(torch.Size([2 * args.n]))

# The splitting ratio of training set, validation set, testing set is 0.7:0.15:0.15
train_size = args.n
valid_size = int(args.n * 0.5)
test_size = 2 * args.n - train_size - valid_size

dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(X, y),
                                                                           [train_size, valid_size, test_size])

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True)

args.loss_criterion = nn.MSELoss(reduction='sum')
args.trueRLCT = (args.output_dim * args.H - args.H ** 2 + args.input_dim * args.H) / 2  # rank r = H for the 'reducedrank_synthetic' dataset


# %%

# define network

class reducedrank(nn.Module):
    def __init__(self, input_dim, output_dim, H):
        super(reducedrank, self).__init__()
        self.fc1 = nn.Linear(input_dim, H, bias=False)
        self.fc2 = nn.Linear(H, output_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# %%

# get B inverse temperatures
args.betas = 1 / np.linspace(np.log(args.n) / args.betasbegin, np.log(args.n) / args.betasend, args.numbetas)
# args.betas = 1 / np.linspace(1 / args.betasbegin, 1 / args.betasend, args.numbetas)


# %%

# define loss function that is specific to anchor point and inverse temperature beta

def custom_loss(model, target, output, beta):

    # returns ||y-\hat y||^2_2 + \sigma_eps^2/beta*\sigma_{prior}^2  ||theta-\hat theta||^2_2
    # anchor_dist = Normal(0.0, args.prior_std)

    wd = torch.tensor(0.)
    for p in model.parameters():
        # anchor = anchor_dist.sample(p.shape)
        # wd += ((p - anchor) ** 2).sum()
        wd += (p ** 2).sum()

    wd_factor = torch.tensor(((args.y_std/args.prior_std)**2)/beta)
    return args.loss_criterion(target, output)/args.batchsize + wd_factor*wd


# %%

# train ensemble

def train(beta):
# return ensemble-average of nL_n(w) = -\sum_{i=1}^n \log p(y_i|x_i,w) = \sum_i (y_i-f(x_i,w))^2/ 2\sigma_eps^2
#     wd_factor = ((args.y_std/args.prior_std)**2)/beta

    model = reducedrank(args.input_dim, args.output_dim, args.H)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=wd_factor)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    wholex = train_loader.dataset[:][0]
    wholey = train_loader.dataset[:][1]

    for epoch in range(1, args.epochs + 1):

        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):

            output = model(data)
            loss = custom_loss(model, target, output, beta)
            # loss = args.loss_criterion(target, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                output = model(wholex)
                eval_loss = custom_loss(model, wholey, output, beta)
                # eval_loss = args.loss_criterion(wholey, output)
                print('Epoch {}: average loss on training {}'.format(epoch, eval_loss/args.n))

    final_output = model(wholex)
    return args.loss_criterion(wholey, final_output).detach()/ (2 * (args.y_std ** 2))
    # return ((wholey - final_output) ** 2).sum() / (2 * (args.y_std ** 2))


nll = np.empty((args.numbetas, args.R))

for beta_index in range(0, args.numbetas):

    beta = args.betas[beta_index]

    for r in range(0, args.R):
        print('Training {}/{} ensemble at {}/{} inverse temp'.format(r + 1, args.R, beta_index + 1, args.numbetas))
        nll[beta_index, r] = train(beta)

# %%

# average nll array over r

ols_model = OLS(np.mean(nll, 1), add_constant(1 / args.betas)).fit()
ols_intercept_estimate = ols_model.params[0]
RLCT_estimate = ols_model.params[1]
print('RLCT estimate: {}'.format(RLCT_estimate))
print('true RLCT: {}'.format(args.trueRLCT))

plt.scatter(1 / args.betas, np.mean(nll, 1), label='nll beta')
plt.plot(1 / args.betas, ols_intercept_estimate + RLCT_estimate * 1 / args.betas, 'b-', label='ols')
plt.show()