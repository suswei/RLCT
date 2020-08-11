import argparse
import numpy as np
import os
from numpy.linalg import inv

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

from matplotlib import pyplot as plt


# TODO: currently only supports realizable reduced rank regression, need to add realizable tanh

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='RLCT Variational Inference')

    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--batchsize', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--H', type=int, default=5)

    parser.add_argument('--dataset', type=str, choices=['tanh', 'rr'])
    parser.add_argument('--prior-std', type=float, default=50.0)
    parser.add_argument('--y-std', type=float, default=1.0)

    parser.add_argument('--betasbegin', type=float, default=0.01,
                        help='where beta range should begin')
    parser.add_argument('--betasend', type=float, default=2.0,
                        help='where beta range should end')
    parser.add_argument('--numbetas', type=int, default=20,
                        help='how many betas should be swept between betasbegin and betasend')

    parser.add_argument('--R', type=int, default=5)

    parser.add_argument('--MC', type=int, default=1)
    parser.add_argument('--taskid', type=int, default=1)

    args = parser.parse_args()

    # %%
    if args.dataset == 'rr':

        args.output_dim = 6
        args.input_dim = 6
        args.H0 = 3

        # args.a_params = torch.transpose(
        #     torch.cat((torch.eye(args.H), torch.ones([args.H, args.input_dim - args.H], dtype=torch.float32)), 1), 0,
        #     1)  # input_dim * H
        # args.b_params = torch.eye(args.output_dim)

        a = Normal(0.0, 1.0)
        args.a_params = 0.2 * a.sample((args.H0, args.input_dim))
        b = Normal(0.0, 1.0)
        args.b_params = 0.2 * b.sample((args.output_dim, args.H0))
        m = MultivariateNormal(torch.zeros(args.input_dim), torch.eye(
            args.input_dim))  # the input_dim=output_dim + 3, output_dim = H (the number of hidden units)
        X = 3.0 * m.sample(torch.Size([2 * args.n]))

        mean = torch.matmul(torch.matmul(X, args.b_params), args.a_params)
        y_rv = MultivariateNormal(torch.zeros(args.output_dim), torch.eye(args.output_dim))
        y = mean + args.y_std * y_rv.sample(torch.Size([2 * args.n]))

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
        args.trueRLCT = (2 * (args.H + args.H0) * (args.input_dim + args.output_dim) - (args.input_dim - args.output_dim) * (args.input_dim - args.output_dim) - (args.H + args.H0) * (args.H + args.H0)) / 8;
        if ((args.input_dim + args.output_dim + args.H + args.H0) % 2 == 1):
            args.trueRLCT = (2 * (args.H + args.H0) * (args.input_dim + args.output_dim) - (args.input_dim - args.output_dim) * (args.input_dim - args.output_dim) - (args.H + args.H0) * (args.H + args.H0) + 1) / 8;
        if (args.H < args.H0):
            args.trueRLCT = args.H * (args.input_dim + args.output_dim - args.H) / 2;
        if (args.input_dim + args.output_dim) < (args.H + args.H0):
            args.trueRLCT = args.input_dim*args.output_dim/2
        # args.trueRLCT = (args.output_dim * args.H - args.H ** 2 + args.input_dim * args.H) / 2  # rank r = H for the 'reducedrank_synthetic' dataset

    elif args.dataset == 'tanh':
        # generate features X from unif(-1,1)

        m = Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
        X = m.sample(torch.Size([2 * args.n]))

        # generate target from N(0,1) i.e. tanh network with zero layers
        # w = {(a_m,b_m)}_{m=1}^p, p(y|x,w) = N(f(x,w),1) where f(x,w) = \sum_{m=1}^p a_m tanh(b_m x)

        y_rv = Normal(0.0,
                      args.y_std)  # torch.distributions.normal.Normal(loc, scale) where scale is standard deviation
        y = y_rv.sample(torch.Size([2 * args.n, 1]))

        # The splitting ratio of training set, validation set, testing set is 0.7:0.15:0.15

        train_size = args.n
        valid_size = int(args.n * 0.5)
        test_size = 2 * args.n - train_size - valid_size
        dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(X, y),
                                                                                   [train_size, valid_size, test_size])

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True)

        args.input_dim = X.shape[1]
        args.output_dim = y.shape[1]

        args.loss_criterion = nn.MSELoss(reduction='sum')
        max_integer = int(np.sqrt(args.H))
        args.trueRLCT = (args.H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)

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

    class tanh(nn.Module):
        def __init__(self, input_dim, output_dim, H):
            super(tanh, self).__init__()
            self.fc1 = nn.Linear(input_dim, H, bias=False)
            self.fc2 = nn.Linear(H, output_dim, bias=False)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    args.w_dim = (args.input_dim + args.output_dim) * args.H

    # TODO: is the log n scale really necessary?
    # get B inverse temperatures
    # args.betas = 1 / np.linspace(np.log(args.n) / args.betasbegin, np.log(args.n) / args.betasend, args.numbetas)
    args.betas = 1 / np.linspace(1 / args.betasbegin, 1 / args.betasend, args.numbetas)
    # args.betas = np.linspace(args.betasbegin, args.betasend, args.numbetas)/np.log(args.n)
    # args.recip = np.linspace(0.1,args.numbetas,args.numbetas) #1/beta
    # args.betas = 1/args.recip
    # args.betas = np.linspace(args.betasbegin, args.betasend, args.numbetas)

    # TODO: set automatically?
    # args.prior_std = np.sqrt(args.w_dim * (args.y_std ** 2) * np.log(args.n) / (args.betasbegin * args.n))
    # # args.prior_std = np.sqrt(args.w_dim / (args.betasbegin * args.n))
    # args.prior_std = 0.1
    # print('prior std auto set to {}'.format(args.prior_std))

    # %%
    # %%

    # define loss function that is specific to anchor point and inverse temperature beta

    def custom_loss(model, target, output, beta, wd_factor):

        # TODO: what's the justification for using anchors?
        # returns ||y-\hat y||^2_2 + \sigma_eps^2/beta*\sigma_{prior}^2  ||theta-\hat theta||^2_2
        # anchor_dist = Normal(0.0, args.prior_std)

        wd = torch.tensor(0.)
        for p in model.parameters():
            # anchor = anchor_dist.sample(p.shape)
            # wd += ((p - anchor) ** 2).sum()
            wd += (p ** 2).sum()

        # wd_factor = torch.tensor(((args.y_std/args.prior_std)**2))
        # print('model fit portion {}'.format(beta * args.loss_criterion(target, output) / (args.batchsize)))
        # print('weight decay portion {}'.format(wd / ((args.prior_std ** 2) * args.n)))
        # return beta * args.loss_criterion(target, output) / (2 * args.batchsize) + wd / (
        #             2 * (args.prior_std ** 2) * args.n)
        criterion = nn.MSELoss(reduction='none')
        all_loss = beta * criterion(target, output) / ((args.y_std ** 2) * args.batchsize) + wd_factor * wd / (
                    (args.prior_std ** 2) * args.n)
        sum_loss = all_loss.sum()
        firstsample_loss = all_loss[0,:].sum()
        return sum_loss, firstsample_loss

        # return args.loss_criterion(target, output) / ((args.y_std ** 2) * args.batchsize) + wd / ((args.prior_std ** 2) * args.n)

    # %%

    # train ensemble

    def train(beta, wd_factor, epochs):
        # return ensemble-average of nL_n(w) = -\sum_{i=1}^n \log p(y_i|x_i,w) = \sum_i (y_i-f(x_i,w))^2/ 2\sigma_eps^2
        #     wd_factor = ((args.y_std/args.prior_std)**2)/beta

        if args.dataset == 'rr':
            model = reducedrank(args.input_dim, args.output_dim, args.H)
        elif args.dataset == 'tanh':
            model = tanh(args.input_dim, args.output_dim, args.H)

        # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=wd_factor)
        # TODO: how to scale lr automatically so it doesn't explode, does it include beta or not?
        # lr = 0.01*args.batchsize / (beta * args.n)

        lr = 0.001 * args.batchsize / (args.n)
        optimizer = optim.SGD(model.parameters(), lr=lr)


        wholex = train_loader.dataset[:][0]
        wholey = train_loader.dataset[:][1]

        for epoch in range(1, epochs + 1):

            model.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                output = model(data)
                loss,_ = custom_loss(model, target, output, beta, wd_factor)
                # loss = args.loss_criterion(target, output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # for p in model.parameters():
                #     batch_grad = p.grad.detach()

            #     model.eval()
            #     with torch.no_grad():
            #         optimizer.zero_grad()
            #         firstsample_loss.backward()
            #         for p in model.parameters():
            #             firstsamplegrad = p.grad.detach()
            #
            #         diff = firstsamplegrad - batch_grad
            #         C = torch.matmul(diff,diff.T)
            #
            # lr = args.batchsize / args.n * (args.w_dim / torch.trace(C))
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr

            if epoch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    output = model(wholex)
                    eval_loss, _ = custom_loss(model, wholey, output, beta, wd_factor)
                    # eval_loss = args.loss_criterion(wholey, output)
                    print('Epoch {}: total loss on training {}, negloglik {}'.format(epoch,
                                                                                     eval_loss,
                                                                                     args.loss_criterion(wholey,output).detach() / ( 2 * (args.y_std ** 2))))

        final_output = model(wholex)
        return ((wholey - final_output) ** 2).sum() / (2 * (args.y_std ** 2))

    # calculate nll of mle
    beta = 1.0
    RHS = train(beta, 0.0, 1000)

    # calculate WBIC at 1/log n
    beta = 1.0/np.log(args.n)
    nll = np.empty((1, args.R))
    for r in range(0, args.R):
        print('Training {}/{} ensemble'.format(r + 1,args.R))
        nll[0, r] = train(beta, 1.0, 1000)
    LHS = np.mean(nll, 1)

    RLCT_estimate = (torch.tensor(LHS)-RHS.detach())/np.log(args.n)

    print('RLCT estimate: {}'.format(RLCT_estimate))
    print('true RLCT: {}'.format(args.trueRLCT))

    # robust ls fit
    # regr = ElasticNet(random_state=0, fit_intercept=True, alpha=0.5)
    # regr.fit((1 / args.betas).reshape(args.numbetas, 1), np.mean(nll, 1))
    # robust_intercept_estimate = regr.intercept_
    # # slope_estimate = min(regr.coef_[0],args.w_dim/2)
    # robust_slope_estimate = regr.coef_[0]

    path = './taskid{}'.format(args.taskid)
    if not os.path.exists(path):
        os.makedirs(path)

    args_dict = vars(args)
    print(args_dict)
    torch.save(args_dict, '{}/mc{}_config.pt'.format(path, args.MC))



if __name__ == "__main__":
    main()