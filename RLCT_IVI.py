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
import copy
import pickle
import itertools

import models
from dataset_factory import get_dataset_by_id


def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    # TODO: count parameters automatically
    if args.network == 'logistic':
        if args.dataset in ('MNIST-binary', 'iris-binary', 'breastcancer-binary', 'lr_synthetic'):
            w_dim = (input_dim + 1)
        elif args.dataset == 'MNIST':
            w_dim = (input_dim + 1) * 9 / 2
    else:
        w_dim = count_parameters(model) * (output_dim - 1) / output_dim

    return model, w_dim


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


# TODO: should test module be used for logging purposes?
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


# Draw w^* from generator G and evaluate nL_n(w^*) on train_loader
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

            # TODO: this block has to be currently manually designed for each model
            data, target = load_minibatch(args, data, target)
            output = torch.mm(data, A.reshape(w_dim - 1, 1)) + b
            output_cat_zero = torch.cat((output, torch.zeros(data.shape[0], 1)), 1)
            logsoftmax_output = F.log_softmax(output_cat_zero, dim=1)
            # input to nll_loss should be log-probabilities of each class. input has to be a Tensor of size either (minibatch, C)
            nll_new = F.nll_loss(logsoftmax_output, target, reduction="sum")

            nll = np.append(nll, np.array(nll_new.detach().cpu().numpy()))

    return nll.sum()


class Discriminator(nn.Module):
    """
    input layer dim = w_dim, output layer dim = 1
    first layer Linear(w_dim, n_hidden_D) followed by ReLU
    num_hidden_layers_D of Linear(n_hidden_D, n_hidden_D) followed by ReLU
    final layer Linear(n_hidden_D, 1)
    """

    def __init__(self, w_dim, n_hidden_D, num_hidden_layers_D=2):
        super().__init__()

        self.enc_sizes = np.concatenate(
            ([w_dim], np.repeat(n_hidden_D, num_hidden_layers_D + 1), [1])).tolist()
        blocks = [[nn.Linear(in_f, out_f), nn.ReLU()]
                  for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        blocks = list(itertools.chain(*blocks))
        del blocks[-1]  # remove the last ReLu, don't need it in output layer

        self.net = nn.Sequential(*blocks)

    def forward(self, w):
        return self.net(w)


class Generator(nn.Module):
    """
    input layer dim = epsilon_dim, output layer dim = w_dim
    first layer Linear(epsilon_dim, n_hidden_G) followed by ReLU
    num_hidden_layers_G of Linear(n_hidden_G, n_hidden_G) followed by ReLU
    final layer Linear(n_hidden_G, w_dim)
    """

    def __init__(self, epsilon_dim, w_dim, n_hidden_G, num_hidden_layers_G=2):
        super().__init__()

        self.enc_sizes = np.concatenate(([epsilon_dim],np.repeat(n_hidden_G,num_hidden_layers_G+1),[w_dim])).tolist()
        blocks = [ [nn.Linear(in_f,out_f),nn.ReLU()]
                       for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        blocks = list(itertools.chain(*blocks))
        del blocks[-1] # remove the last ReLu, don't need it in output layer

        self.net = nn.Sequential(*blocks)

    def forward(self, epsilon):

        return self.net(epsilon)


def randn(shape, device):

    if device == False:
        # return torch.randn(*shape).to(device)
        return torch.randn(*shape)
    else:
        return torch.cuda.FloatTensor(*shape).normal_()


def lsfit_lambda(temperedNLL_perMC_perBeta, betas):

    ols_model = OLS(temperedNLL_perMC_perBeta, add_constant(1 / betas)).fit()

    ols_resid = ols_model.resid
    res_fit = OLS(list(ols_resid[1:]), list(ols_resid[:-1])).fit()
    rho = res_fit.params

    order = toeplitz(np.arange(betas.__len__()))
    sigma = rho ** order

    gls_model = GLS(temperedNLL_perMC_perBeta, add_constant(1 / betas), sigma=sigma).fit()

    return ols_model.params[1], gls_model.params[1]


# Approximate inference estimate of E_w^\beta [nL_n(w)]:  1/R \sum_{r=1}^R nL_n(w_r^*)
def approxinf_expected_betanll(train_loader, test_loader, input_dim, output_dim, args, beta):

    model, w_dim = retrieve_model(args,input_dim,output_dim)
    args.epsilon_dim = w_dim
    args.epsilon_mc = args.batchsize  # TODO: overwriting args parser input

    # instantiate generator and discriminator
    G_initial = Generator(args.epsilon_dim, w_dim, args.n_hidden_G, args.num_hidden_layers_G)  # G = Generator(args.epsilon_dim, w_dim).to(args.cuda)
    D_initial = Discriminator(w_dim, args.n_hidden_D)  # D = Discriminator(w_dim).to(args.cuda)
    G = copy.deepcopy(G_initial)
    print(G)
    D = copy.deepcopy(D_initial)

    # optimizers
    opt_primal = optim.Adam(
        G.parameters(),
        lr=args.lr_primal)
    opt_dual = optim.Adam(
        D.parameters(),
        lr=args.lr_dual)

    # pretrain discriminator
    for epoch in range(args.pretrainDepochs):

        w_sampled_from_prior = randn((args.batchsize, w_dim), args.cuda)
        eps = randn((args.batchsize, args.epsilon_dim), args.cuda)
        w_sampled_from_G = G(eps)
        loss_dual = torch.mean(-F.logsigmoid(D(w_sampled_from_G)) - F.logsigmoid(-D(w_sampled_from_prior)))

        loss_dual.backward()
        opt_dual.step()
        G.zero_grad()
        D.zero_grad()

    # train discriminator and generator together
    for epoch in range(args.epochs):

        correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            # opt discriminator more than generator
            for discriminator_epoch in range(args.trainDepochs):

                w_sampled_from_prior = randn((args.epsilon_mc, w_dim), args.cuda) # TODO: add more options for prior besides hardcoding Gaussian prior
                eps = randn((args.epsilon_mc, args.epsilon_dim), args.cuda)
                w_sampled_from_G = G(eps)
                loss_dual = torch.mean(-F.logsigmoid(D(w_sampled_from_G)) - F.logsigmoid(-D(w_sampled_from_prior)))
                loss_dual.backward()
                opt_dual.step()
                G.zero_grad()
                D.zero_grad()

            data, target = load_minibatch(args, data, target)

            # opt generator
            eps = randn((args.epsilon_mc, args.epsilon_dim), args.cuda)
            w_sampled_from_G = G(eps)

            # for fixed minibatch of size b, reconstr_err approximates
            # E_\epsilon frac{1}{b} \sum_{i=b}^b -log p(y_i|x_i, G(epsilon)) with args.epsilon_mc realisations
            reconstr_err = 0
            for i in range(args.epsilon_mc):  # loop over rows of w_sampled_from_G corresponding to different epsilons

                # TODO: this block has to be currently manually designed for each model
                A = w_sampled_from_G[i, 0:(w_dim-1)]
                b = w_sampled_from_G[i, w_dim-1]
                output = torch.mm(data, A.reshape(w_dim-1, 1))+b
                output_cat_zero = torch.cat((output,torch.zeros(data.shape[0],1)),1)
                logsoftmax_output = F.log_softmax(output_cat_zero, dim=1)
                # input to nll_loss should be log-probabilities of each class. input has to be a Tensor of size (minibatch, C)
                reconstr_err += F.nll_loss(logsoftmax_output, target, reduction="mean")

            loss_primal = reconstr_err/args.epsilon_mc + torch.mean(D(w_sampled_from_G))/(beta*args.n)
            loss_primal.backward(retain_graph=True)
            opt_primal.step()
            G.zero_grad()
            D.zero_grad()

            # minibatch logging on args.log_interval
            pred = logsoftmax_output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            if batch_idx % args.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss primal: {:.6f}\tLoss dual: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss_primal.data.item(), loss_dual.data.item()))

        # epoch logging
        print('\nTrain set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))

    my_list = range(args.R)
    num_cores = 1  # multiprocessing.cpu_count()

    # approxinf_nlls equals array [nL_n(w_1^*),\ldots, nL_n(w_R^*)] where w^* is drawn from generator G
    approxinf_nlls = Parallel(n_jobs=num_cores, verbose=0)(delayed(approxinf_nll)(i, train_loader, G, model, args) for i in my_list)

    return np.asarray(approxinf_nlls).mean(), np.asarray(approxinf_nlls)


# Thm 4 of Watanabe's WBIC: E_w^\beta[nL_n(w)] = nL_n(w_0) + \lambda/\beta + U_n \sqrt(\lambda/\beta)
def lambda_thm4(betas, args, kwargs):

    RLCT_estimates_GLS = np.empty(0)
    RLCT_estimates_OLS = np.empty(0)

    for mc in range(0, args.MCs):

        print('Starting MC {}'.format(mc))
        # draw new training-testing split
        train_loader, test_loader, input_dim, output_dim = get_dataset_by_id(args, kwargs)

        temperedNLL_perMC_perBeta = np.empty(0)
        for beta in betas:
            temp, _ = approxinf_expected_betanll(train_loader, test_loader, input_dim, output_dim, args, beta)
            temperedNLL_perMC_perBeta = np.append(temperedNLL_perMC_perBeta, temp)

        # least squares fit for lambda
        ols, gls = lsfit_lambda(temperedNLL_perMC_perBeta, betas)
        RLCT_estimates_GLS = np.append(RLCT_estimates_GLS, gls)
        RLCT_estimates_OLS = np.append(RLCT_estimates_OLS, ols)

        plt.scatter(1 / betas, temperedNLL_perMC_perBeta)
        plt.title("Thm 4, one MC realisation: hat lambda = {:.2f}, true lambda = {:.2f}".format(gls, args.w_dim/2))
        plt.xlabel("1/beta")
        plt.ylabel("implicit VI estimate of E^beta_w [nL_n(w)]")
        plt.show()

        print("RLCT GLS: {}".format(RLCT_estimates_GLS))

        if args.wandb_on:
            import wandb
            wandb.run.summary["running RLCT OLS"] = RLCT_estimates_OLS
            wandb.run.summary["running RLCT GLS"] = RLCT_estimates_GLS

        print('Finishing MC {}'.format(mc))

    # return array of RLCT estimates, length args.MCs
    return RLCT_estimates_OLS, RLCT_estimates_GLS


# apply E_{D_n} to Theorem 4 of Watanabe's WBIC: E_{D_n} E_w^\beta[nL_n(w)] = E_{D_n} nL_n(w_0) + \lambda/\beta
def lambda_thm4average(betas, args, kwargs):

    temperedNLL_perBeta = np.empty(0)

    for beta in betas:

        print('Starting beta {}'.format(beta))

        temperedNLL_perMC_perBeta = np.empty(0)

        for mc in range(0, args.MCs):

            # draw new training-testing split
            train_loader, test_loader, input_dim, output_dim = get_dataset_by_id(args, kwargs)
            temp, _ = approxinf_expected_betanll(train_loader,
                                              test_loader,
                                              input_dim,
                                              output_dim,
                                              args,
                                              beta)
            temperedNLL_perMC_perBeta = np.append(temperedNLL_perMC_perBeta, temp)

        temperedNLL_perBeta = np.append(temperedNLL_perBeta, temperedNLL_perMC_perBeta.mean())

        print('Finishing beta {}'.format(beta))


    plt.scatter(1 / betas, temperedNLL_perMC_perBeta)
    plt.title("multiple MC realisation")
    plt.xlabel("1/beta")
    plt.ylabel("implicit VI estimate of E_{D_n} E^beta_w [nL_n(w)]")
    plt.show()
    RLCT_estimate_OLS, RLCT_estimate_GLS = lsfit_lambda(temperedNLL_perMC_perBeta, betas)

    # each RLCT estimate is one elment array
    return RLCT_estimate_OLS, RLCT_estimate_GLS


def lambda_cor3(betas, args, kwargs):

    RLCT_estimates = np.empty(0)

    for mc in range(0, args.MCs):

        # draw new training-testing split
        train_loader, test_loader, input_dim, output_dim = get_dataset_by_id(args, kwargs)

        lambdas_beta1 = np.empty(0)
        for beta in betas:

            beta1 = beta
            beta2 = beta+0.05/np.log(args.n)
            _, nlls = approxinf_expected_betanll(train_loader, test_loader, input_dim, output_dim, args, beta1)

            lambda_beta1 = (nlls.mean() - (nlls * np.exp(-(beta2 - beta1) * nlls)).mean() / (np.exp(-(beta2 - beta1) * nlls)).mean()) / (1 / beta1 - 1 / beta2)
            lambdas_beta1 = np.append(lambdas_beta1, lambda_beta1)
            RLCT_estimates = np.append(RLCT_estimates, lambdas_beta1.mean())


        print('MC: {} RLCT estimate: {:.2f}'.format(mc, lambdas_beta1.mean()))

    return RLCT_estimates


def main():
    random.seed()

    # Training settings
    parser = argparse.ArgumentParser(description='RLCT Implicit Variational Inference')
    parser.add_argument('--dataset',
                        type=str,
                        default='lr_synthetic',
                        help='dataset name from dataset_factory.py (default: )',
                        choices=['iris-binary', 'breastcancer-binary', 'MNIST-binary', 'MNIST','lr_synthetic'])

    parser.add_argument('--syntheticsamplesize',
                        type=int,
                        default=60000,
                        help='sample size of synthetic dataset')

    parser.add_argument('--network',
                        type=str,
                        default='logistic',
                        help='name of network in models.py (default: logistic)',
                        choices=['FFrelu','CNN','logistic'])

    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        metavar='N',
                        help='number of epochs to train (default: 100)')

    parser.add_argument('--batchsize',
                        type=int,
                        default=10,
                        metavar='N',
                        help='input batch size for training (default: 10)')

    parser.add_argument('--betasbegin',
                        type=float,
                        default=0.1,
                        help='where beta range should begin')

    parser.add_argument('--betasend',
                        type=float,
                        default=2,
                        help='where beta range should end')

    parser.add_argument('--betalogscale',
                        type=str,
                        default='true',
                        help='true if beta should be on 1/log n scale (default: true)',
                        choices=['true','false'])

    parser.add_argument('--n_hidden_D',
                        type=int,
                        default=256,
                        help='number of hidden units in discriminator D')

    parser.add_argument('--num_hidden_layers_D',
                        type = int,
                        default=2,
                        help='number of hidden layers in discriminatro D')

    parser.add_argument('--n_hidden_G',
                        type=int,
                        default=256,
                        help='number of hidden units in generator G')

    parser.add_argument('--num_hidden_layers_G',
                        type = int,
                        default = 2,
                        help = 'number of hidden layers in generator G')

    parser.add_argument('--lambda_asymptotic',
                        type=str,
                        default='thm4',
                        help='which asymptotic characterisation of lambda to use',
                        choices=['thm4', 'thm4_average', 'cor3'])

    parser.add_argument('--pretrainDepochs', 
                        type=int, 
                        default=2,
                        help='number of epochs to pretrain discriminator')
    
    parser.add_argument('--trainDepochs', 
                        type=int, 
                        default=2,
                        help='number of epochs to train discriminator for each minibatch update of generator')
    
    parser.add_argument('--dpower',
                        type=float,
                        default=2/5,
                        help='set dimension of model to n^dpower')
    
    # as high as possible
    # parser.add_argument('--epsilon_mc', type=int, default=10,
    #                     help='number of draws for estimating E_\epsilon')
    parser.add_argument('--numbetas',
                        type=int, 
                        default=20,
                        help='how many betas should be swept between betasbegin and betasend')

    parser.add_argument('--MCs',
                        type=int,
                        default=100,
                        help='number of times to split into train-test')

    parser.add_argument('--R',
                        type=int,
                        default=100,
                        help='number of MC draws from approximate posterior (default:100)')

    parser.add_argument('--wandb_on',
                        action="store_true",
                        help='use wandb to log experiment')

    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--lr_primal',
                        type=float,
                        default=1e-4,
                        metavar='LR',
                        help='primal learning rate (default: 0.01)')

    parser.add_argument('--lr_dual',
                        type=float,
                        default=1e-4,
                        metavar='LR',
                        help='dual learning rate (default: 0.01)')

    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum',
                        type=float,
                        default=0.5,
                        metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval',
                        type=int,
                        default=10,
                        metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--prior',
                        type=str,
                        default='gaussian',
                        metavar='P',
                        help='prior used on model parameters (default: gaussian)',
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

    # set true network weights for synthetic dataset
    if args.dataset == 'lr_synthetic':
        input_dim = int(np.power(args.syntheticsamplesize, args.dpower))
        args.w_0 = torch.randn(input_dim,1)
        args.b = torch.randn(1)

    # draw a training-testing split just to get some necessary parameters
    train_loader, test_loader, input_dim, output_dim = get_dataset_by_id(args, kwargs)
    args.n = len(train_loader.dataset)

    # retrieve model
    model, w_dim = retrieve_model(args, input_dim, output_dim)
    args.w_dim = w_dim
    print(model)
    print('(number of parameters)/2: {}'.format(w_dim/2))

    # sweep betas
    betas = 1/np.linspace(1/args.betasbegin, 1/args.betasend, args.numbetas)
    if args.betalogscale == 'true':
        betas = 1/np.linspace(np.log(args.n)/args.betasbegin, np.log(args.n)/args.betasend, args.numbetas)

    if args.lambda_asymptotic == 'thm4':

        RLCT_estimates_OLS, RLCT_estimates_GLS = lambda_thm4(betas, args, kwargs)
        results = dict({
            "d on 2": w_dim / 2,
            "RLCT_estimates (OLS)": RLCT_estimates_OLS,
            "RLCT_estimates (GLS)": RLCT_estimates_GLS,
            "abs deviation of average RLCT estimate (OLS) from d on 2": np.abs(RLCT_estimates_OLS.mean() - w_dim / 2),
            "abs deviation of average RLCT estimate (GLS) from d on 2": np.abs(RLCT_estimates_GLS.mean() - w_dim / 2)
        })

    elif args.lambda_asymptotic == 'thm4_average':

        RLCT_estimate_OLS, RLCT_estimate_GLS = lambda_thm4average(betas, args, kwargs)
        results = dict({
            "d on 2": w_dim/2,
            "RLCT_estimate (OLS)": RLCT_estimate_OLS,
            "RLCT_estimate (GLS)": RLCT_estimate_GLS,
            "abs deviation of RLCT estimate (OLS) from d on 2": np.abs(RLCT_estimate_OLS - w_dim / 2),
            "abs deviation of RLCT estimate (GLS) from d on 2": np.abs(RLCT_estimate_GLS - w_dim / 2)
        })

    elif args.lambda_asymptotic == 'cor3':

        RLCT_estimates = lambda_cor3(betas, args, kwargs)
        results = dict({
            "d on 2": w_dim/2,
            "RLCT estimates (one per training set)": RLCT_estimates,
            "average RLCT estimate": RLCT_estimates.mean(),
            "abs deviation of RLCT estimate from d on 2": np.abs(RLCT_estimates.mean() - w_dim/2)})


    print(results)
    if args.wandb_on:
        wandb.log(results)
        f = open("results.pkl", "wb")
        pickle.dump(results, f)
        f.close()
        wandb.save("results.pkl")

if __name__ == "__main__":
    main()


