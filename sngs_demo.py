
import torch.nn.functional as F

import numpy as np
from numpy.linalg import inv

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from scipy.stats import levy_stable

from matplotlib import pyplot as plt


### good result for lr, seeing iissues at H ~ 160, more sensitivity to temperature here

# self.n = 1000
# self.batchsize = 50
# self.epochs = 1000
#
# self.H = 20
# self.eps = 1
#
# self.betasbegin = 0.5
# self.betasend = 2.0
# self.numbetas = 20

# self.R = 50
# self.stepsizedecay = 1.0

###

# self.dataset = 'tanh'
# self.n = 1000
# self.batchsize = 100
# self.epochs = 1000
#
# self.H = 256
# self.eps = 1.0
#
# self.betasbegin = 5.0
# self.betasend = 10.0
# self.numbetas = 20
#
# self.R = 50
# self.stepsizedecay = 0.3


class args():

    def __init__(self):
        self.dataset = 'rr'
        self.n = 1000
        self.batchsize = 50
        self.epochs = 2000

        self.H = 10
        self.eps = 1e-5
        self.stepsizedecay = 0.9

        self.betasbegin = 1.0
        self.betasend = 2.0
        self.numbetas = 20

        self.R = 5

        self.method = 'simsekli'


args = args()


def plot_energy(inverse_temp, nll):

    numbetas = inverse_temp.shape[0]
    design_x = np.vstack((np.ones(numbetas), 1 / inverse_temp)).T
    # TODO: use median rather than mean?
    design_y = np.nanmedian(nll, 1)
    design_y = design_y[:, np.newaxis]
    fit = inv(design_x.T.dot(design_x)).dot(design_x.T).dot(design_y)
    ols_intercept_estimate = fit[0][0]
    RLCT_estimate = fit[1][0]

    # TODO: use median rather than mean?
    plt.scatter(1 / inverse_temp, np.nanmedian(nll, 1), label='nll beta')
    plt.plot(1 / inverse_temp, ols_intercept_estimate + RLCT_estimate / inverse_temp, 'b-', label='ols')
    plt.title(
        "d/2 = {} , true lambda = {:.1f}, est lambda = {:.1f}".format(args.w_dim / 2, args.trueRLCT, RLCT_estimate),
        fontsize=8)
    plt.xlabel("1/beta", fontsize=8)
    plt.ylabel("E^beta_w [nL_n(w)]", fontsize=8)
    plt.legend()
    plt.show()


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


if args.dataset == 'lr':

    unit_normal = Normal(0.0, 1.0)
    args.w0 = 0.2 * unit_normal.sample((args.H,1))
    args.b0 = unit_normal.sample([1])
    X = torch.randn(args.n, args.H)
    affine = torch.mm(X, args.w0) + args.b0
    m = torch.distributions.bernoulli.Bernoulli(torch.sigmoid(affine))
    y = m.sample()

    criterion_sum = nn.BCELoss(reduction='sum')
    criterion_mean = nn.BCELoss(reduction='mean')
    baseline_nll = criterion_sum(torch.sigmoid(affine), y)

    args.input_dim = args.H
    args.output_dim = 1
    args.w_dim = args.H+1
    args.trueRLCT = args.w_dim/2

elif args.dataset == 'tanh':

    m = Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
    X = m.sample(torch.Size([args.n]))

    # generate target from N(0,1)
    y_rv = Normal(0.0, 0.1)  # torch.distributions.normal.Normal(loc, scale) where scale is standard deviation
    y = y_rv.sample(torch.Size([args.n, 1]))

    args.input_dim = X.shape[1]
    args.output_dim = y.shape[1]
    criterion_sum = nn.MSELoss(reduction='sum')
    baseline_nll = criterion_sum(torch.zeros_like(y), y)

    args.w_dim = (args.input_dim + args.output_dim) * args.H
    max_integer = int(np.sqrt(args.H))
    args.trueRLCT = (args.H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)

elif args.dataset == 'tanh_nontrivial':

    m = Normal(0.0,2.0)
    args.input_dim = 3
    X = m.sample(torch.Size([args.n, args.input_dim]))

    # generate target from N(0,1)
    args.output_dim = 3
    H0 = 1
    y_std = 0.1
    y_rv = Normal(0.0, 0.1)  # torch.distributions.normal.Normal(loc, scale) where scale is standard deviation
    a0 = Normal(0.0, 1.0)
    a_params = a0.sample((args.input_dim,H0))
    b0 = Normal(0.0, 1.0)
    b_params = b0.sample((H0, args.output_dim))
    true_mean = torch.matmul(torch.tanh(torch.matmul(X, a_params)), b_params)

    # true_model = tanh(args.input_dim, args.output_dim, 1)
    # true_mean = true_model(X)
    y = true_mean + y_std * y_rv.sample(torch.Size([args.n, args.output_dim]))

    criterion_sum = nn.MSELoss(reduction='sum')
    baseline_nll = criterion_sum(true_mean, y)

    args.w_dim = (args.input_dim + args.output_dim) * args.H
    max_integer = int(np.sqrt(args.H))
    args.trueRLCT = (args.H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)

elif args.dataset == 'rr':

    args.output_dim = 6
    args.input_dim = 6
    args.H0 = 3
    args.y_std = 0.1

    a = Normal(0.0, 1.0)
    args.a_params = 0.2 * a.sample((args.input_dim, args.H0))
    b = Normal(0.0, 1.0)
    args.b_params = 0.2 * b.sample((args.H0, args.output_dim))
    m = MultivariateNormal(torch.zeros(args.input_dim), torch.eye(
        args.input_dim))  # the input_dim=output_dim + 3, output_dim = H (the number of hidden units)
    X = 3.0 * m.sample(torch.Size([args.n]))

    true_mean = torch.matmul(torch.matmul(X, args.a_params), args.b_params)
    y_rv = MultivariateNormal(torch.zeros(args.output_dim), torch.eye(args.output_dim))
    y = true_mean + args.y_std * y_rv.sample(torch.Size([args.n]))


    criterion_sum = nn.MSELoss(reduction='sum')
    baseline_nll = criterion_sum(true_mean, y)


    args.w_dim = (args.input_dim + args.output_dim) * args.H

    cond1 = (args.output_dim+args.H0) <= (args.input_dim+args.H)
    cond2 = (args.input_dim+args.H0) <= (args.output_dim + args.H)
    cond3 = (args.H + args.H0) <= (args.input_dim+args.output_dim)
    if cond1 and cond2 and cond3:

        args.trueRLCT = (2 * (args.H + args.H0) * (args.input_dim + args.output_dim)
                         - (args.input_dim - args.output_dim) * (args.input_dim - args.output_dim)
                         - (args.H + args.H0) * (args.H + args.H0)) / 8 #case 1a in Aoygai
        if ((args.input_dim + args.output_dim + args.H + args.H0) % 2 == 1): # case 1b in Aoyagi
            args.trueRLCT = (2 * (args.H + args.H0) * (args.input_dim + args.output_dim) - (
                        args.input_dim - args.output_dim) * (args.input_dim - args.output_dim) - (args.H + args.H0) * (
                                         args.H + args.H0) + 1) / 8
    if (args.input_dim + args.H) < (args.output_dim + args.H0): # case 2 in Aoyagi
        args.trueRLCT = (args.H * (args.input_dim -args.H0) + args.output_dim*args.H0) / 2
    if (args.output_dim + args.H) < (args.input_dim + args.H0): # case 3 in Aoyagi
        args.trueRLCT = (args.H * (args.output_dim -args.H0) + args.input_dim*args.H0) / 2
    if (args.input_dim + args.output_dim) < (args.H + args.H0): # case 4 in Aoyagi
        args.trueRLCT = (args.H * (args.input_dim -args.H0) + args.output_dim*args.H0) / 2

dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(X, y),[args.n, 0, 0])
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True)


nll = np.empty((args.numbetas, args.R))
# evenly spaced 1/betaend to 1/betabegin, low temp to high temp, should see increase in nll
inverse_temp = np.flip(1 / np.linspace(1 / args.betasbegin, 1 / args.betasend, args.numbetas))


for beta_index in range(0, args.numbetas):

    beta = inverse_temp[beta_index]

    for r in range(0, args.R):

        if args.dataset == 'lr':
            model = logistic(args.input_dim)
        elif args.dataset in ['tanh', 'tanh_nontrivial']:
            model = tanh(args.input_dim,args.output_dim,args.H)
        elif args.dataset == 'rr':
            model = reducedrank(args.input_dim,args.output_dim,args.H)

        print('Training {}/{} ensemble at {}/{} temp'.format(r + 1, args.R, beta_index + 1, args.numbetas))

        # according to Mandt, this is the ideal constant stepsize for approximating the posterior, assumed to be Gaussian
        if args.method == 'mandt':
            stepsize = args.eps * 2 * args.batchsize/(beta*args.n)

        # Training
        for epoch in range(1, args.epochs):

            stepsize = (1e-9/(epoch+1)) ** 0.6
            for p in model.parameters():

                if args.method == 'mandt':
                    output = model(X)

                    loss = criterion_sum(output, y) / args.n
                    loss.backward()

                    nrv = Normal(0.0, 1.0/np.sqrt(args.batchsize))
                    blah = nrv.sample(p.shape)
                    p.data -= stepsize*(p.grad+blah)

                elif args.method == 'simsekli':

                    output = model(X)

                    loss = criterion_sum(output,y)
                    loss.backward()

                    alpha = 1.5

                    num = torch.tensor([alpha -1])
                    denom = torch.tensor( [alpha/2] )
                    c_alpha = torch.exp(torch.lgamma(num))/ (torch.exp(torch.lgamma(denom)) ** 2)
                    # TODO: wait a minute, this doesn't depend on temperature...
                    symmetric_alpha_stable_draw = levy_stable.rvs(alpha,0,size=(p.shape[0],p.shape[1]))
                    p.data -= stepsize * c_alpha * beta* p.grad - (stepsize ** (1/alpha)) * symmetric_alpha_stable_draw
            # PyTorch stuffs
            model.zero_grad()

            if (epoch+1) % 200 == 0:
                print('Epoch {}: training nll {}, baseline nll {}'.format(epoch, criterion_sum(model(X), y), baseline_nll))
                # stepsize = args.stepsizedecay*args.eps

        # Record nll
        nll[beta_index, r] = criterion_sum(model(X), y)

    if beta_index > 0:
        plot_energy(inverse_temp[0:beta_index + 1], nll[0:beta_index + 1, :])

plot_energy(inverse_temp, nll)
