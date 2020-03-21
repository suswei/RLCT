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
from collections import OrderedDict

import models
from dataset_factory import get_dataset_by_id


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test(epoch, test_loader, model, args, verbose=False):

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

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


# approximate E^beta_w[nL_n(w)] where the expectation is approximated by drawing w^* once using generator G and finding nL_n(w^*)
def rsamples_nll(r, train_loader, G, model, args):

    G.eval()
    eps = randn((1, args.epsilon_dim), args.cuda)
    w_sampled_from_G = G(eps)

    new_state_dict = OrderedDict()
    begin = 0
    for (k, v) in model.state_dict().items():
        end = begin + v.nelement()
        idx = torch.arange(begin, end)
        temp = w_sampled_from_G.index_select(1, idx)
        new_state_dict.update({k: temp.view(v.shape)})
        begin = end
    model.load_state_dict(new_state_dict, strict=True)


    nll = np.empty(0)
    for batch_idx, (data, target) in enumerate(train_loader):

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

        output = model(data)
        nll = np.append(nll, np.array(F.nll_loss(output, target, reduction="sum").detach().cpu().numpy()))

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


def IVI_temperedNLL_perMC_perBeta(train_loader, test_loader, input_dim, output_dim, args, beta):

    # retrieve model
    if args.network == 'CNN':
        model = models.CNN(output_dim=output_dim)
    if args.network == 'logistic':
        model = models.LogisticRegression(input_dim=input_dim, output_dim=output_dim)
    if args.network == 'FFrelu':
        model = models.FFrelu(input_dim=input_dim, output_dim=output_dim)

    w_dim = count_parameters(model)
    args.epsilon_dim = w_dim*2

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

    # pretrain discriminator for some time before starting iterative process
    for epoch in range(50):

        w_sampled_from_prior = randn((args.batchsize, w_dim), args.cuda)
        eps = randn((args.batchsize, args.epsilon_dim), args.cuda)
        w_sampled_from_G = G(eps)
        loss_dual = torch.mean(-F.logsigmoid(D(w_sampled_from_G)) - F.logsigmoid(-D(w_sampled_from_prior)))

        loss_dual.backward()
        opt_dual.step()
        G.zero_grad()
        D.zero_grad()


    # train together
    for epoch in range(args.epochs):

        train_loss = 0
        correct = 0
        ELBO = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            # opt discriminator more than generator
            for epoch in range(50):
                w_sampled_from_prior = randn((args.batchsize, w_dim), args.cuda)
                eps = randn((args.batchsize, args.epsilon_dim), args.cuda)
                w_sampled_from_G = G(eps)
                loss_dual = torch.mean(-F.logsigmoid(D(w_sampled_from_G)) - F.logsigmoid(-D(w_sampled_from_prior)))

                loss_dual.backward()
                opt_dual.step()
                G.zero_grad()
                D.zero_grad()

            # opt generator
            eps = randn((args.batchsize, args.epsilon_dim), args.cuda)
            w_sampled_from_G = G(eps)

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

            # for fixed minibatch reconstr_err approximates E_\epsilon frac{1}{m} \sum_{i=1}^m -log p(y_i|x_i, G(epsilon)) with multiple epsilon realisation
            reconstr_err = 0
            for i in range(args.batchsize): # loop over rows of w_sampled_from_G corresponding to different epsilons
                new_state_dict = OrderedDict()
                begin = 0
                for (k,v) in model.state_dict().items():
                    end = begin + v.nelement()
                    idx = torch.arange(begin,end)
                    temp = w_sampled_from_G[i,:].index_select(0,idx)
                    new_state_dict.update({k : temp.view(v.shape)})
                    begin = end
                model.load_state_dict(new_state_dict, strict=True)

                output = model(data)
                reconstr_err += beta * F.nll_loss(output, target, reduction="mean")

            loss_primal = reconstr_err/args.batchsize + torch.mean(D(w_sampled_from_G))/(beta*args.n)
            loss_primal.backward(retain_graph=True)
            opt_primal.step()
            G.zero_grad()
            D.zero_grad()

    # draws R samples {w_1,\ldots,w_R} from r_\theta^\beta (var_model) and returns \frac{1}{R} \sum_{i=1}^R [nL_n(w_i}]
    my_list = range(args.R)
    num_cores = 1  # multiprocessing.cpu_count()
    nlls = Parallel(n_jobs=num_cores, verbose=0)(delayed(rsamples_nll)(i, train_loader, G, model, args) for i in my_list)
    temperednlls_perMC_perBeta = np.asarray(nlls)

    return temperednlls_perMC_perBeta.mean()


# TODO: this is only for categorical prediction at the moment, relies on nll_loss in pytorch. Should generalise eventually
# estimating Bayes RLCT based on variational inference
def main():
    random.seed()

    # Training settings
    parser = argparse.ArgumentParser(description='RLCT Implicit Variational Inference')
    # crucial parameters
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='dataset name from dataset_factory.py (default:MNIST)')
    parser.add_argument('--network', type=str, default='CNN',
                        help='name of network in models.py (default:CNN)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batchsize', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--betasbegin', type=float, default=0.1,
                        help='where beta range should begin')
    parser.add_argument('--betasend', type=float, default=2,
                        help='where beta range should end')
    parser.add_argument('--betalogscale', type=str, default='true',
                        help='true if beta should be on 1/log n scale')
    parser.add_argument('--fit_lambda_over_average', type=str, default='false',
                        help='true lambda should be fit after averaging tempered nlls')
    # as high as possible
    parser.add_argument('--bl', type=int, default=50,
                        help='how many betas should be swept between betasbegin and betasend')
    parser.add_argument('--MCs', type=int, default=50,
                        help='number of times to split into train-test')
    parser.add_argument('--R', type=int, default=50,
                        help='number of MC draws from approximate posterior q (default:)')
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
    if args.network == 'CNN':
        model = models.CNN(output_dim=output_dim)
    if args.network == 'logistic':
        model = models.LogisticRegression(input_dim=input_dim, output_dim=output_dim)
    if args.network == 'FFrelu':
        model = models.FFrelu(input_dim=input_dim, output_dim=output_dim)
    print(model)

    # d/2
    # TODO: doesn't look like I'm counting the number of parameters correctly. For binary lgoistic regression on MNIST-binary, the number of parameters should be 784+1, so d/2 is 785/2
    if args.network == 'logistic':
        if args.dataset in ('MNIST-binary', 'iris-binary', 'breastcancer-binary'):
            d_on_2 = (input_dim + 1) // 2
        elif args.dataset == 'MNIST':
            d_on_2 = (input_dim + 1) * 9 // 2
    else:
        d_on_2 = count_parameters(model) * (output_dim - 1) / output_dim // 2

    print('(number of parameters)/2: {}'.format(d_on_2))

    # sweep betas
    betas = np.linspace(args.betasbegin, args.betasend, args.bl)
    if args.betalogscale == 'true':
        betas = np.linspace(args.betasbegin / np.log(args.n), args.betasend / np.log(args.n), args.bl)

    # Use E_{D_n} E_w^\beta[nL_n(w)] = E_{D_n} nL_n(w_0) + \lambda/\beta
    if args.fit_lambda_over_average == 'true':
        temperedNLL_perBeta = np.empty(0)
        for beta in betas:
            temperedNLL_perMC_perBeta = np.empty(0)
            for mc in range(0, args.MCs):
                # draw new training-testing split
                train_loader, test_loader, input_dim, output_dim = get_dataset_by_id(args, kwargs)
                temp = IVI_temperedNLL_perMC_perBeta(train_loader, test_loader, input_dim, output_dim, args,
                                                          kwargs,
                                                          prior_parameters, beta)
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

                temp = IVI_temperedNLL_perMC_perBeta(train_loader, test_loader, input_dim, output_dim, args, beta)
                temperedNLL_perMC_perBeta = np.append(temperedNLL_perMC_perBeta, temp)

            plt.scatter(1/betas,temperedNLL_perMC_perBeta)
            plt.show()

            # GLS fit for lambda
            ols_model = OLS(temperedNLL_perMC_perBeta, add_constant(1 / betas)).fit()
            RLCT_estimates_OLS = np.append(RLCT_estimates_OLS, ols_model.params[1])

            ols_resid = ols_model.resid
            res_fit = OLS(list(ols_resid[1:]), list(ols_resid[:-1])).fit()
            rho = res_fit.params

            order = toeplitz(np.arange(args.bl))
            sigma = rho ** order

            gls_model = GLS(temperedNLL_perMC_perBeta, add_constant(1 / betas), sigma=sigma)
            gls_results = gls_model.fit()

            RLCT_estimates_GLS = np.append(RLCT_estimates_GLS, gls_results.params[1])
            print("RLCT_estimates_GLS: {}".format(RLCT_estimates_GLS))
            if args.wandb_on:
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


