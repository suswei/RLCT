from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from joblib import Parallel, delayed
import random
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.tools import add_constant
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt

import models
from dataset_factory import get_dataset_by_id


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_minibatch(args,data,target):

    if args.dataset == 'MNIST-binary':
        for ind, y_val in enumerate(target):
            target[ind] = 0 if y_val < 5 else 1

    if args.cuda:
        data, target = data.cuda(), target.cuda()

    if args.dataset in ('MNIST', 'MNIST-binary'):
        if args.network == 'CNN':
            data, target = Variable(data), Variable(target)
        else:
            data, target = Variable(data.view(-1, 28 * 28)), Variable(target)
    else:
        data, target = Variable(data), Variable(target)

    return data, target


def retrieve_model(args,input_dim,output_dim):

    # retrieve model
    if args.network == 'CNN':
        model = models.CNN(output_dim=output_dim)
        print('Error: implicit VI currently only supports logistic regression')
    if args.network == 'logistic':
        model = models.LogisticRegression(input_dim=input_dim, output_dim=output_dim)
    if args.network == 'FFrelu':
        model = models.FFrelu(input_dim=input_dim, output_dim=output_dim)
        print('Error: implicit VI currently only supports logistic regression')

    # TODO: how to count parameters automatically?
    if args.network == 'logistic':
        if args.dataset in ('MNIST-binary', 'iris-binary', 'breastcancer-binary'):
            w_dim = (input_dim + 1)
        elif args.dataset == 'MNIST':
            w_dim = (input_dim + 1) * 9 / 2
    else:
        w_dim = count_parameters(model) * (output_dim - 1) / output_dim

    return model, w_dim


def test(epoch, test_loader, model, args, verbose=False):

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            data,target = load_minibatch(args, data, target)
            output = model(data)
            test_loss += F.nll_loss(output, target).data.item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


# Draw w^* from generator G
# Evaluate nL_n(w^*) on train_loader  
def approxinf_nll(r, train_loader, G, model, args):

    G.eval()
    with torch.no_grad():
        eps = randn((1, args.epsilon_dim), args.cuda)
        w_sampled_from_G = G(eps)
        w_dim = w_sampled_from_G.shape[1]

        A = w_sampled_from_G[0, 0:(w_dim - 1)]
        b = w_sampled_from_G[0, w_dim - 1]

        nll = np.empty(0)
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = load_minibatch(args, data, target)
            output = torch.mm(data, A.reshape(w_dim - 1, 1)) + b
            output_cat_zero = torch.cat((output, torch.zeros(data.shape[0], 1)), 1)
            logsoftmax_output = F.log_softmax(output_cat_zero, dim=1)
            # input to nll_loss should be log-probabilities of each class. input has to be a Tensor of size either (minibatch, C)
            nll_new = F.nll_loss(logsoftmax_output, target, reduction="sum")
            nll = np.append(nll, np.array(nll_new.detach().cpu().numpy()))

    return nll.sum()

# TODO: discriminator hidden layer dims should not be hardcoded
# D(w) maps to real line
class Discriminator(nn.Module):

    def __init__(self, w_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(w_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, w):

        return self.net(w)

# TODO: discriminator hidden layer dims should not be hardcoded
# Given epsilon input, Generator outputs w
class Generator(nn.Module):

    def __init__(self, epsilon_dim, w_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(epsilon_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, w_dim)
        )

    def forward(self, epsilon):
        return self.net(epsilon)


def randn(shape, device):

    if device == False:
        # return torch.randn(*shape).to(device)
        return torch.randn(*shape)
    else:
        return torch.cuda.FloatTensor(*shape).normal_()


def approxinf_expected_betanll(train_loader, test_loader, input_dim, output_dim, args, beta):

    model, w_dim = retrieve_model(args,input_dim,output_dim)
    args.epsilon_dim = w_dim

    # instantiate generator and discriminator
    # G = Generator(args.epsilon_dim, w_dim).to(args.cuda)
    # D = Discriminator(w_dim).to(args.cuda)
    G = Generator(args.epsilon_dim, w_dim)
    D = Discriminator(w_dim)

    opt_primal = optim.Adam(
        G.parameters(),
        lr=args.lr_primal)
    opt_dual = optim.Adam(
        D.parameters(),
        lr=args.lr_dual)

    G.train()

    # pretrain discriminator
    for epoch in range(5):

        w_sampled_from_prior = randn((args.batchsize, w_dim), args.cuda)
        eps = randn((args.batchsize, args.epsilon_dim), args.cuda)
        w_sampled_from_G = G(eps)
        loss_dual = torch.mean(-F.logsigmoid(D(w_sampled_from_G)) - F.logsigmoid(-D(w_sampled_from_prior)))

        loss_dual.backward()
        opt_dual.step()
        G.zero_grad()
        D.zero_grad()

    epsilon_mc = 5
    # train discriminator and generator together
    for epoch in range(args.epochs):

        train_loss = 0
        correct = 0
        ELBO = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            # opt discriminator more than generator
            for discriminator_epoch in range(5):

                w_sampled_from_prior = randn((epsilon_mc, w_dim), args.cuda)
                eps = randn((epsilon_mc, args.epsilon_dim), args.cuda)
                w_sampled_from_G = G(eps)
                loss_dual = torch.mean(-F.logsigmoid(D(w_sampled_from_G)) - F.logsigmoid(-D(w_sampled_from_prior)))
                loss_dual.backward()
                opt_dual.step()
                G.zero_grad()
                D.zero_grad()

            data, target = load_minibatch(args, data, target)

            # opt generator
            eps = randn((epsilon_mc, args.epsilon_dim), args.cuda)
            w_sampled_from_G = G(eps)

            # for fixed minibatch of size b, reconstr_err approximates
            # E_\epsilon frac{1}{b} \sum_{i=b}^b -log p(y_i|x_i, G(epsilon)) with epsilon_mc realisations
            reconstr_err = 0
            for i in range(epsilon_mc):  # loop over rows of w_sampled_from_G corresponding to different epsilons
                A = w_sampled_from_G[i, 0:(w_dim-1)]
                b = w_sampled_from_G[i, w_dim-1]
                output = torch.mm(data, A.reshape(w_dim-1, 1))+b
                output_cat_zero = torch.cat((output,torch.zeros(data.shape[0],1)),1)
                logsoftmax_output = F.log_softmax(output_cat_zero, dim=1)
                # input to nll_loss should be log-probabilities of each class. input has to be a Tensor of size either (minibatch, C)
                reconstr_err += F.nll_loss(logsoftmax_output, target, reduction="mean")

            loss_primal = reconstr_err/epsilon_mc + torch.mean(D(w_sampled_from_G))/(beta*args.n)
            loss_primal.backward(retain_graph=True)
            opt_primal.step()
            G.zero_grad()
            D.zero_grad()

            with torch.no_grad():
                pred = logsoftmax_output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data).cpu().sum()
                if batch_idx % args.log_interval == 0:
                    print(
                        'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss primal: {:.6f}\tLoss dual: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), loss_primal.data.item(), loss_dual.data.item()))

        print('\nTrain set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))

    my_list = range(args.R)
    num_cores = 1  # multiprocessing.cpu_count()
    # return array [nL_n(w_1^*),\ldots, nL_n(w_R^*)] where w^* is drawn from generator G
    approxinf_nlls = Parallel(n_jobs=num_cores, verbose=0)(delayed(approxinf_nll)(i, train_loader, G, model, args) for i in my_list)

    # Approximate inference estimate of E_w^\beta [nL_n(w)]:  1/R \sum_{r=1}^R nL_n(w_r^*)
    return np.asarray(approxinf_nlls).mean()


# TODO: this is only for categorical prediction at the moment, relies on nll_loss in pytorch. Should generalise eventually
# estimating Bayes RLCT based on variational inference
def main():
    random.seed()

    # Training settings
    parser = argparse.ArgumentParser(description='RLCT Implicit Variational Inference')
    # crucial parameters
    parser.add_argument('--dataset', type=str, default='breastcancer-binary',
                        help='dataset name from dataset_factory.py (default: breastcancer-binary)')
    parser.add_argument('--network', type=str, default='logistic',
                        help='name of network in models.py (default: logistic)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--batchsize', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--betasbegin', type=float, default=0.1,
                        help='where beta range should begin')
    parser.add_argument('--betasend', type=float, default=2,
                        help='where beta range should end')
    parser.add_argument('--betalogscale', type=str, default='true',
                        help='true if beta should be on 1/log n scale (default: true)')
    parser.add_argument('--fit_lambda_over_average', type=str, default='false',
                        help='true lambda should be fit after averaging tempered nlls (default: false)')
    # as high as possible
    parser.add_argument('--bl', type=int, default=50,
                        help='how many betas should be swept between betasbegin and betasend')
    parser.add_argument('--MCs', type=int, default=100,
                        help='number of times to split into train-test')
    parser.add_argument('--R', type=int, default=50,
                        help='number of MC draws from approximate posterior (default:50)')
    # not so crucial parameters can accept defaults
    parser.add_argument('--wandb_on', action="store_true",
                        help='use wandb to log experiment')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr_primal', type=float, default=1e-4, metavar='LR',
                        help='primal learning rate (default: 0.01)')
    parser.add_argument('--lr_dual', type=float, default=1e-4, metavar='LR',
                        help='dual learning rate (default: 0.01)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--prior', type=str, default='gaussian', metavar='P',
                        help='prior used (default: gaussian)',
                        choices=['gaussian', 'mixtgauss', 'conjugate', 'conjugate_known_mean'])
    args = parser.parse_args()
    print(vars(args))

    if args.wandb_on:
        import wandb
        wandb.init(project="RLCT", entity="unimelb_rlct")
        wandb.config.update(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("args.cuda is " + str(args.cuda))

    # setting up prior parameters
    prior_parameters = {}
    if args.prior != 'gaussian':
        prior_parameters['n_mc_samples'] = 10
    if args.prior == 'mixtgauss':
        prior_parameters['sigma_1'] = 0.02
        prior_parameters['sigma_2'] = 0.2
        prior_parameters['pi'] = 0.5
    if args.prior == 'conjugate':
        prior_parameters['mu_0'] = 0.
        prior_parameters['kappa_0'] = 3.
        prior_parameters['alpha_0'] = .5
        prior_parameters['beta_0'] = .5
    if args.prior == 'conjugate_known_mean':
        prior_parameters['alpha_0'] = .5
        prior_parameters['beta_0'] = .5
        prior_parameters['mean'] = 0.

    # Daniel
    # torch.manual_seed(args.seed)
    # if args.cuda:
    #    torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # draw a training-testing split just to get some necessary parameters
    train_loader, test_loader, input_dim, output_dim = get_dataset_by_id(args, kwargs)
    args.n = len(train_loader.dataset)

    # retrieve model
    model, w_dim = retrieve_model(args, input_dim, output_dim)
    print(model)
    print('(number of parameters)/2: {}'.format(w_dim/2))

    # sweep betas
    betas = np.linspace(args.betasbegin, args.betasend, args.bl)
    if args.betalogscale == 'true':
        betas = 1/np.linspace(np.log(args.n)/args.betasbegin, np.log(args.n)/args.betasend, args.bl)

    # Use E_{D_n} E_w^\beta[nL_n(w)] = E_{D_n} nL_n(w_0) + \lambda/\beta
    if args.fit_lambda_over_average == 'true':
        temperedNLL_perBeta = np.empty(0)
        for beta in betas:
            temperedNLL_perMC_perBeta = np.empty(0)
            for mc in range(0, args.MCs):
                # draw new training-testing split
                train_loader, test_loader, input_dim, output_dim = get_dataset_by_id(args, kwargs)
                temp = approxinf_expected_betanll(train_loader,
                                                  test_loader,
                                                  input_dim,
                                                  output_dim,
                                                  args,
                                                  kwargs,
                                                  prior_parameters,
                                                  beta)
                temperedNLL_perMC_perBeta = np.append(temperedNLL_perMC_perBeta, temp)
            temperedNLL_perBeta = np.append(temperedNLL_perBeta, temperedNLL_perMC_perBeta.mean())

        # GLS fit for lambda
        ols_model = OLS(temperedNLL_perBeta, add_constant(1 / betas)).fit()
        RLCT_estimate_OLS = ols_model.params[1]

        ols_resid = ols_model.resid
        res_fit = OLS(list(ols_resid[1:]), list(ols_resid[:-1])).fit()
        rho = res_fit.params

        order = toeplitz(np.arange(args.bl))
        sigma = rho ** order

        gls_model = GLS(temperedNLL_perBeta, add_constant(1 / betas), sigma=sigma)
        gls_results = gls_model.fit()
        RLCT_estimate_GLS = gls_results.params[1]

    # Use E_w^\beta[nL_n(w)] = nL_n(w_0) + \lambda/\beta + U_n \sqrt(\lambda/\beta)
    else:

        RLCT_estimates_GLS = np.empty(0)
        RLCT_estimates_OLS = np.empty(0)

        for mc in range(0, args.MCs):

            print('Starting MC {}'.format(mc))
            # draw new training-testing split
            train_loader, test_loader, input_dim, output_dim = get_dataset_by_id(args, kwargs)

            temperedNLL_perMC_perBeta = np.empty(0)
            for beta in betas:

                temp = approxinf_expected_betanll(train_loader, test_loader, input_dim, output_dim, args, beta)
                temperedNLL_perMC_perBeta = np.append(temperedNLL_perMC_perBeta, temp)

            plt.scatter(1/betas,temperedNLL_perMC_perBeta)
            plt.show()

            # GLS fit for lambda
            ols_model = OLS(temperedNLL_perMC_perBeta, add_constant(1 / betas)).fit()
            RLCT_estimates_OLS = np.append(RLCT_estimates_OLS, ols_model.params[1])
            print("RLCT_estimates OLS: {}".format(RLCT_estimates_OLS))

            ols_resid = ols_model.resid
            res_fit = OLS(list(ols_resid[1:]), list(ols_resid[:-1])).fit()
            rho = res_fit.params

            order = toeplitz(np.arange(args.bl))
            sigma = rho ** order

            gls_model = GLS(temperedNLL_perMC_perBeta, add_constant(1 / betas), sigma=sigma)
            gls_results = gls_model.fit()

            RLCT_estimates_GLS = np.append(RLCT_estimates_GLS, gls_results.params[1])
            print("RLCT_estimates GLS: {}".format(RLCT_estimates_GLS))
            if args.wandb_on:
                wandb.run.summary["RLCT_estimate_OLS"] = RLCT_estimates_OLS
                wandb.run.summary["RLCT_estimate_GLS"] = RLCT_estimates_GLS

            print('Finishing MC {}'.format(mc))

        RLCT_estimate_OLS = RLCT_estimates_OLS.mean()
        RLCT_estimate_GLS = RLCT_estimates_GLS.mean()


    results = dict({
        "RLCT_estimate (OLS)": RLCT_estimate_OLS,
        "RLCT_estimate (GLS)": RLCT_estimate_GLS,
        "abs deviation of RLCT estimate (GLS) from d on 2": np.abs(RLCT_estimate_GLS - d_on_2)})

    print(results)
    if args.wandb_on:
        wandb.log(results)


if __name__ == "__main__":
    main()


