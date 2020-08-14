# implements variations of Langevin Monte Carlo
# Mandt: uses optimal constant stepsize with preconditioning for Gaussian-assumed posterior
# Simsekli: FLA uses symmetric alpha stable noise, can recover multimodal posterior more easily

import os
import numpy as np
from numpy.linalg import inv
import argparse

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

def plot_energy(args, inverse_temp, nll, savefigname, savefig=False):

    numbetas = inverse_temp.shape[0]
    design_x = np.vstack((np.ones(numbetas), 1 / inverse_temp)).T
    # TODO: use median rather than mean?
    design_y = np.nanmean(nll, 1)
    design_y = design_y[:, np.newaxis]
    fit = inv(design_x.T.dot(design_x)).dot(design_x.T).dot(design_y)
    ols_intercept_estimate = fit[0][0]
    RLCT_estimate = fit[1][0]

    # TODO: use median rather than mean?
    plt.scatter(1 / inverse_temp, np.nanmean(nll, 1), label='nll beta')
    plt.plot(1 / inverse_temp, ols_intercept_estimate + RLCT_estimate / inverse_temp, 'b-', label='ols')
    plt.title(
        "d/2 = {} , true lambda = {:.1f}, est lambda = {:.1f}".format(args.w_dim / 2, args.trueRLCT, RLCT_estimate),
        fontsize=8)
    plt.xlabel("1/beta", fontsize=8)
    plt.ylabel("E^beta_w [nL_n(w)]", fontsize=8)
    plt.legend()
    if savefig:
        plt.savefig('taskid{}/{}.png'.format(args.taskid,savefigname))
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


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='RLCT Variational Inference')

    # dataset
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--dataset', type=str, choices=['tanh','rr'],default='rr')
    # data generating parameters
    parser.add_argument('--rr',  nargs='*', type=float, help='input_dim, output_dim, H0, y_std, a_std, b_std', default=[20, 20, 3, 0.1, 0.2, 0.2])
    parser.add_argument('--tanh',  nargs='*', type=float, help='unif start, unif end, y_std', default=[10, 10, 3, 0.1])
    parser.add_argument('--tanh-nontrivial',  nargs='*', type=int, help='input_dim, output_dim, H0, x_std, y_std', default=[10, 10, 3, 1.0, 0.1])

    # model and training
    parser.add_argument('--H', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=10) # TODO: the way we are using Mandt does not actually require batchsize
    parser.add_argument('--epochs', type=int, default=2000)

    # inverse temps
    parser.add_argument('--betasbegin', type=float, default=2.0,
                        help='where beta range should begin')
    parser.add_argument('--betasend', type=float, default=3.0,
                        help='where beta range should end')
    parser.add_argument('--numbetas', type=int, default=20,
                        help='how many betas should be swept between betasbegin and betasend')

    parser.add_argument('--R', type=int, default=50)

    parser.add_argument('--method', default='simsekli', type=str, choices=['simsekli','mandt'])
    # method specific parameters
    parser.add_argument('--mandt-params', nargs='*', type=float, help='stepsize factor, stepsize decay', default=[1e-8,0.9])
    parser.add_argument('--simsekli-params', nargs='*', type=float, help='alpha, eta, and gamma in (eta/t)^gamma', default=[1.3,1e-8,0.6])

    parser.add_argument('--taskid',type=int, default=1)

    args = parser.parse_args()

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

        unif_start = args.tanh[0]
        unif_end = args.tanh[1]
        args.y_std = args.tanh[2]

        m = Uniform(torch.tensor([unif_start]), torch.tensor([unif_end]))
        X = m.sample(torch.Size([args.n]))

        # generate target from N(0,1)
        y_rv = Normal(0.0, 0.1)  # torch.distributions.normal.Normal(loc, scale) where scale is standard deviation
        y = args.y_std*y_rv.sample(torch.Size([args.n, 1]))

        args.input_dim = X.shape[1]
        args.output_dim = y.shape[1]
        criterion_sum = nn.MSELoss(reduction='sum')
        baseline_nll = criterion_sum(torch.zeros_like(y), y)

        args.w_dim = (args.input_dim + args.output_dim) * args.H
        max_integer = int(np.sqrt(args.H))
        args.trueRLCT = (args.H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)

    elif args.dataset == 'tanh_nontrivial':

        args.input_dim = args.tanh_nontrivial[0]
        args.output_dim = args.tanh_nontrivial[1]
        args.H0 = args.tanh_nontrivial[2]
        args.x_std = args.tanh_nontrivial[3]
        args.y_std = args.tanh_nontrivial[4]

        # generate input
        m = Normal(0.0, args.x_std)
        X = m.sample(torch.Size([args.n, args.input_dim]))

        # true mean
        a0 = Normal(0.0, 1.0)
        a_params = a0.sample((args.input_dim,args.H0))
        b0 = Normal(0.0, 1.0)
        b_params = b0.sample((args.H0, args.output_dim))
        true_mean = torch.matmul(torch.tanh(torch.matmul(X, a_params)), b_params)

        # generate target from N(true_mean,y_std)
        y_rv = Normal(0.0, 1.0)  # torch.distributions.normal.Normal(loc, scale) where scale is standard deviation
        y = true_mean + args.y_std * y_rv.sample(torch.Size([args.n, args.output_dim]))

        criterion_sum = nn.MSELoss(reduction='sum')
        baseline_nll = criterion_sum(true_mean, y)

        args.w_dim = (args.input_dim + args.output_dim) * args.H
        max_integer = int(np.sqrt(args.H))
        args.trueRLCT = (args.H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)

    elif args.dataset == 'rr':

        args.input_dim = args.rr[0]
        args.output_dim = args.rr[1]
        args.H0 = args.rr[2]
        args.y_std = args.rr[3]
        args.a_std = args.rr[4]
        args.b_std = args.rr[5]

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

        # determining theoretical RLCT
        cond1 = (args.output_dim+args.H0) <= (args.input_dim+args.H)
        cond2 = (args.input_dim+args.H0) <= (args.output_dim + args.H)
        cond3 = (args.H + args.H0) <= (args.input_dim+args.output_dim)
        if cond1 and cond2 and cond3:
            args.trueRLCT = (2 * (args.H + args.H0) * (args.input_dim + args.output_dim)
                             - (args.input_dim - args.output_dim) * (args.input_dim - args.output_dim)
                             - (args.H + args.H0) * (args.H + args.H0)) / 8 #case 1a in Aoygai
            if (args.input_dim + args.output_dim + args.H + args.H0) % 2 == 1: # case 1b in Aoyagi
                args.trueRLCT = (2 * (args.H + args.H0) * (args.input_dim + args.output_dim) - (
                            args.input_dim - args.output_dim) * (args.input_dim - args.output_dim) - (args.H + args.H0) * (
                                             args.H + args.H0) + 1) / 8
        if (args.input_dim + args.H) < (args.output_dim + args.H0): # case 2 in Aoyagi
            args.trueRLCT = (args.H * (args.input_dim - args.H0) + args.output_dim*args.H0) / 2
        if (args.output_dim + args.H) < (args.input_dim + args.H0): # case 3 in Aoyagi
            args.trueRLCT = (args.H * (args.output_dim - args.H0) + args.input_dim*args.H0) / 2
        if (args.input_dim + args.output_dim) < (args.H + args.H0): # case 4 in Aoyagi
            args.trueRLCT = (args.H * (args.input_dim - args.H0) + args.output_dim*args.H0) / 2
        # For practical use, the case of input_dim >> H and output_dim >>H are considered, so Case (4) does not occur.

    dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(X, y),[args.n, 0, 0])
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True)


    nll = np.empty((args.numbetas, args.R))
    # evenly spaced 1/betaend to 1/betabegin, low temp to high temp, should see increase in nll
    inverse_temp = np.flip(1 / np.linspace(1 / args.betasbegin, 1 / args.betasend, args.numbetas),axis=0)

    path = './taskid{}'.format(args.taskid)
    if not os.path.exists(path):
        os.makedirs(path)

    args_dict = vars(args)
    print(args_dict)
    torch.save(args_dict, '{}/config.pt'.format(path))

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

            # Training
            for epoch in range(1, args.epochs):

                # according to Mandt, this is the ideal constant stepsize for approximating the posterior, assumed to be Gaussian
                if args.method == 'mandt':

                    stepsize_factor = args.mandt_params[0]
                    stepsize = stepsize_factor * 2 * args.batchsize / (beta * args.n)

                elif args.method == 'simsekli':

                    alpha = args.simsekli_params[0]
                    stepsize = (args.simsekli_params[1]/(epoch+1)) ** args.simsekli_params[2]
                    num = torch.tensor([alpha - 1])
                    denom = torch.tensor([alpha / 2])
                    c_alpha = torch.exp(torch.lgamma(num)) / (torch.exp(torch.lgamma(denom)) ** 2)


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
                        loss = criterion_sum(output, y)
                        loss.backward()

                        # TODO: wait a minute, this doesn't depend on temperature...
                        symmetric_alpha_stable_draw = levy_stable.rvs(alpha, 0, size=(p.shape[0], p.shape[1]))
                        p.data -= stepsize * c_alpha * beta * p.grad - (stepsize ** (1/alpha)) * symmetric_alpha_stable_draw

                model.zero_grad()

                if (epoch+1) % 200 == 0:
                    print('Epoch {}: training nll {:.2f}, baseline nll {:.2f}'.format(epoch, criterion_sum(model(X), y), baseline_nll))
                    if args.method == 'mandt':
                        stepsize = args.mandt_params[1] * stepsize

            # Record nll
            nll[beta_index, r] = criterion_sum(model(X), y)

        temp = nll[beta_index,:]
        plt.hist(temp[~np.isnan(temp)])
        plt.title('nLn(w) at inverse temp {}'.format(beta))
        plt.savefig('taskid{}/nllhist_invtemp{:.2f}.png'.format(args.taskid,beta))
        plt.show()

        if beta_index > 0:
            plot_energy(args, inverse_temp[0:beta_index + 1], nll[0:beta_index + 1, :],savefig=True,savefigname='lsfit_upto_beta{:.2f}'.format(beta_index))

    plot_energy(args, inverse_temp, nll, savefig=True, savefigname='lsfit')
    results = {'invtemp': inverse_temp, 'nll': nll}
    torch.save(results, 'results.pt')

if __name__ == "__main__":
    main()