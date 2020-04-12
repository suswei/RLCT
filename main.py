from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import random
from matplotlib import pyplot as plt
import copy
import pickle
import itertools
import math
import pyvarinf
import models
from dataset_factory import get_dataset_by_id
from RLCT_helper import retrieve_model, load_minibatch, randn, lsfit_lambda



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


# TODO: this needs to be put into the pyvarinf framework as Mingming has demonstrated in main_ivi and ivi.py
def train_implicitVI(train_loader, valid_loader, args, mc, beta, betas):

    # instantiate generator and discriminator
    G_initial = Generator(args.epsilon_dim, args.w_dim, args.n_hidden_G,
                          args.num_hidden_layers_G)  # G = Generator(args.epsilon_dim, w_dim).to(args.cuda)
    D_initial = Discriminator(args.w_dim, args.n_hidden_D)  # D = Discriminator(w_dim).to(args.cuda)
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
    scheduler_G = ReduceLROnPlateau(opt_primal, mode='min', factor=0.1, patience=3, verbose=True)

    # pretrain discriminator
    for epoch in range(args.pretrainDepochs):
        w_sampled_from_prior = randn((args.batchsize, args.w_dim), args.cuda)
        eps = randn((args.batchsize, args.epsilon_dim), args.cuda)
        w_sampled_from_G = G(eps)
        loss_dual = torch.mean(-F.logsigmoid(D(w_sampled_from_G)) - F.logsigmoid(-D(w_sampled_from_prior)))

        loss_dual.backward()
        opt_dual.step()
        G.zero_grad()
        D.zero_grad()

    if args.dataset in ['3layertanh_synthetic','reducedrank_synthetic']:
        train_loss, valid_loss = [], []

    # train discriminator and generator together
    for epoch in range(args.epochs):

        D.train()

        if args.dataset == 'lr_synthetic':
            correct = 0
        elif args.dataset in ['3layertanh_synthetic', 'reducedrank_synthetic']:
            training_sum_se = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            # opt discriminator more than generator
            for discriminator_epoch in range(args.trainDepochs):
                w_sampled_from_prior = randn((args.epsilon_mc, args.w_dim),
                                             args.cuda)  # TODO: add more options for prior besides hardcoding Gaussian prior
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

                # TODO: (HUI) this block has to be currently manually designed for each model
                if args.dataset == 'lr_synthetic':
                    A = w_sampled_from_G[i, 0:(args.w_dim - 1)]
                    b = w_sampled_from_G[i, args.w_dim - 1]
                    output = torch.mm(data, A.reshape(args.w_dim - 1, 1)) + b
                    output_cat_zero = torch.cat((output, torch.zeros(data.shape[0], 1)), 1)
                    logsoftmax_output = F.log_softmax(output_cat_zero, dim=1)
                    # input to nll_loss should be log-probabilities of each class. input has to be a Tensor of size (minibatch, C)
                    reconstr_err += args.loss(logsoftmax_output, target)
                elif args.dataset in ['3layertanh_synthetic', 'reducedrank_synthetic']:
                    a_params = w_sampled_from_G[i, 0:(args.input_dim * args.H)].reshape(args.input_dim, args.H)
                    b_params = w_sampled_from_G[i, (args.input_dim * args.H):].reshape(args.H, args.output_dim)
                    if args.dataset == '3layertanh_synthetic':
                        output = torch.matmul(torch.tanh(torch.matmul(data, a_params)), b_params)
                    else:
                        output = torch.matmul(torch.matmul(data, a_params), b_params)
                    reconstr_err += args.loss(output, target) #reduction is set to be 'mean' by default

            loss_primal = reconstr_err / args.epsilon_mc + torch.mean(D(w_sampled_from_G)) / (beta * args.n)
            loss_primal.backward(retain_graph=True)
            opt_primal.step()
            G.zero_grad()
            D.zero_grad()

            # minibatch logging on args.log_interval
            if args.dataset == 'lr_synthetic':
                pred = logsoftmax_output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data).cpu().sum()
            elif args.dataset in ['3layertanh_synthetic', 'reducedrank_synthetic']:
                training_sum_se += args.loss(output, target).detach().cpu().numpy()*len(target)
            if batch_idx % args.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss primal: {:.6f}\tLoss dual: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss_primal.data.item(), loss_dual.data.item()))
        if (mc < 10) and (beta==betas[0]) and (args.dataset in ['3layertanh_synthetic', 'reducedrank_synthetic']):
            with torch.no_grad(): #to save memory, no intermediate activations used for activation calculation is stored.
                D.eval()
                valid_sum_se = 0
                for valid_batch_id, (valid_data, valid_target) in enumerate(valid_loader):
                    valid_data, valid_target = load_minibatch(args, valid_data, valid_target)
                    if args.dataset == '3layertanh_synthetic':
                       valid_output = torch.matmul(torch.tanh(torch.matmul(valid_data, a_params)), b_params)
                    else:
                       valid_output = torch.matmul(torch.matmul(valid_data, a_params), b_params)
                    valid_sum_se += args.loss(valid_output, valid_target).detach().cpu().numpy()*len(valid_target)
                valid_loss += [valid_sum_se/len(valid_loader.dataset)+ torch.mean(D(w_sampled_from_G)) / (beta * len(train_loader.dataset))]

                train_sum_se = 0
                for train_batch_id, (train_data, train_target) in enumerate(train_loader):
                    train_data, train_target = load_minibatch(args, train_data, train_target)
                    if args.dataset == '3layertanh_synthetic':
                        train_output = torch.matmul(torch.tanh(torch.matmul(train_data, a_params)), b_params)
                    else:
                        train_output = torch.matmul(torch.matmul(train_data, a_params), b_params)
                    train_sum_se += args.loss(train_output, train_target).detach().cpu().numpy() * len(train_target)
                train_loss += [train_sum_se / len(train_loader.dataset) + torch.mean(D(w_sampled_from_G)) / (beta * len(train_loader.dataset))]

        # epoch logging
        if args.dataset == 'lr_synthetic':
            print('\nTrain set: Accuracy: {}/{} ({:.0f}%)\n'.format(
                correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))
        elif args.dataset in ['3layertanh_synthetic', 'reducedrank_synthetic']:
            print('\nTrain set: MSE: {} \n'.format(training_sum_se/len(train_loader.dataset)))

    if (mc <10) and (beta==betas[0]) and (args.dataset in ['3layertanh_synthetic', 'reducedrank_synthetic']):
        fig = plt.figure(figsize=(10, 7))
        lines1 = plt.plot(list(range(0, args.epochs)), train_loss,
                          list(range(0, args.epochs)), valid_loss)
        plt.legend(('training loss', 'validation loss'), loc='upper right', fontsize=16)
        plt.xlabel('epoch number', fontsize=16)
        plt.title('training validation loss', fontsize=18)
        fig.savefig('./result/training_validation_loss_MC%s.png'%(mc))
        plt.close(fig)

    return G

def train_explicitVI(train_loader, args, beta, verbose=True):


    # retrieve model
    if args.network == 'CNN':
        model = models.CNN(output_dim=args.output_dim)
        print('Error: implicit VI currently only supports logistic regression')
    if args.network == 'logistic':
        model = models.LogisticRegression(input_dim=args.input_dim, output_dim=args.output_dim)
    if args.network == 'FFrelu':
        model = models.FFrelu(input_dim=args.input_dim, output_dim=args.output_dim)
        print('Error: implicit VI currently only supports logistic regression')
    if args.network == 'Tanh':
        model = models.Tanh(input_dim=args.input_dim, output_dim=args.output_dim, H=args.H)
    if args.network == 'ReducedRankRegression':
        model = models.ReducedRankRegression(input_dim=args.input_dim, output_dim=args.output_dim, H=args.H)


    # variationalize model
    var_model_initial = pyvarinf.Variationalize(model)

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
        
    var_model = copy.deepcopy(var_model_initial)
    var_model.set_prior(args.prior, **prior_parameters)
    if args.cuda:
        var_model.cuda()
    optimizer = optim.Adam(var_model.parameters(), lr=args.lr)

    if args.dataset in ['3layertanh_synthetic','reducedrank_synthetic']:
        MSEloss = nn.MSELoss(reduction='mean')

    # train var_model
    for epoch in range(1, args.epochs + 1):

        var_model.train()

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

            optimizer.zero_grad()
            # var_model draw a sample of the network parameter and then applies the network with the sampled weights
            output = var_model(data)
            if args.dataset == 'lr_synthetic':
                loss_error = F.nll_loss(output, target, reduction="mean")
            elif args.dataset in ['3layertanh_synthetic', 'reducedrank_synthetic']:
                loss_error = MSEloss(output, target)
            loss_prior = var_model.prior_loss() / (beta*args.n)
            loss = loss_error + loss_prior  # this is the ELBO
            loss.backward()
            optimizer.step()
            if verbose:
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss error: {:.6f}\tLoss weights: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data.item(), loss_error.data.item(), loss_prior.data.item()))

    return var_model


# TODO: approxinf_nll_implicit and approxinf_nll_explicit should eventually be merged.
# Draw w^* from generator G and evaluate nL_n(w^*) on train_loader
def approxinf_nll_implicit(r, train_loader, G, model, args):

    G.eval()
    with torch.no_grad():
        eps = randn((1, args.epsilon_dim), args.cuda)
        w_sampled_from_G = G(eps)
        w_dim = w_sampled_from_G.shape[1]
        if args.dataset == 'lr_synthetic':
            A = w_sampled_from_G[0, 0:(w_dim - 1)]
            b = w_sampled_from_G[0, w_dim - 1]
        elif args.dataset in ['3layertanh_synthetic', 'reducedrank_synthetic']:
            a_params = w_sampled_from_G[0, 0:(args.input_dim * args.H)].reshape(args.input_dim, args.H)
            b_params = w_sampled_from_G[0, (args.input_dim * args.H):].reshape(args.H, args.output_dim)

        nll = np.empty(0)
        for batch_idx, (data, target) in enumerate(train_loader):

            # TODO: (HUI) this block has to be currently manually designed for each model
            data, target = load_minibatch(args, data, target)
            if args.dataset == 'lr_synthetic':
                output = torch.mm(data, A.reshape(w_dim - 1, 1)) + b
                output_cat_zero = torch.cat((output, torch.zeros(data.shape[0], 1)), 1)
                logsoftmax_output = F.log_softmax(output_cat_zero, dim=1)
                # input to nll_loss should be log-probabilities of each class. input has to be a Tensor of size either (minibatch, C)
                nll_new = args.loss(logsoftmax_output, target)*len(target) #sum
            elif args.dataset == '3layertanh_synthetic':
                output = torch.matmul(torch.tanh(torch.matmul(data, a_params)), b_params)
                nll_new = args.loss(output, target)*len(target) #sum
            elif args.dataset == 'reducedrank_synthetic':
                output = torch.matmul(torch.matmul(data, a_params), b_params)
                nll_new = args.loss(output, target)*len(target) #sum

            nll = np.append(nll, np.array(nll_new.detach().cpu().numpy()))

    return nll.sum()


# w^* is drawn by calling sample.draw(), this function evaluates nL_n(w^*) on train_loader
def approxinf_nll_explicit(r, train_loader, sample, args):

    if args.dataset in ['3layertanh_synthetic','reducedrank_synthetic']:
        MSEloss = nn.MSELoss(reduction='sum')
    nll = np.empty(0)
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = load_minibatch(args, data, target)

        sample.draw()
        output = sample(data)
        if args.dataset == 'lr_synthetic':
            nll = np.append(nll, np.array(F.nll_loss(output, target, reduction="sum").detach().cpu().numpy()))
        elif args.dataset in ['3layertanh_synthetic', 'reducedrank_synthetic']:
            nll = np.append(nll, np.array(MSEloss(output, target).detach().cpu().numpy()))

    return nll.sum()


# Approximate inference estimate of E_w^\beta [nL_n(w)]:  1/R \sum_{r=1}^R nL_n(w_r^*)
def approxinf_nll(train_loader, valid_loader, test_loader, input_dim, output_dim, args, mc, beta, betas):

    model, w_dim = retrieve_model(args,input_dim,output_dim)
    args.w_dim = w_dim
    args.epsilon_dim = w_dim
    args.epsilon_mc = args.batchsize  # TODO: overwriting args parser input

    if args.VItype == 'implicit':

        G = train_implicitVI(train_loader, valid_loader, args, mc, beta, betas)

        my_list = range(args.R)
        num_cores = 1  # multiprocessing.cpu_count()
        # approxinf_nlls equals array [nL_n(w_1^*),\ldots, nL_n(w_R^*)] where w^* is drawn from generator G
        approxinf_nlls = Parallel(n_jobs=num_cores, verbose=0)(
            delayed(approxinf_nll_implicit)(i, train_loader, G, model, args) for i in my_list)

    elif args.VItype == 'explicit':

        var_model = train_explicitVI(train_loader, args, beta, verbose=True)

        # form sample object from variational distribution r
        sample = pyvarinf.Sample(var_model=var_model)
        # draws R samples {w_1,\ldots,w_R} from r_\theta^\beta (var_model) and returns \frac{1}{R} \sum_{i=1}^R [nL_n(w_i}]
        my_list = range(args.R)
        num_cores = 1  # multiprocessing.cpu_count()
        approxinf_nlls = Parallel(n_jobs=num_cores, verbose=0)(
            delayed(approxinf_nll_explicit)(i, train_loader, sample, args) for i in my_list)


    return np.asarray(approxinf_nlls).mean(), np.asarray(approxinf_nlls)


# Thm 4 of Watanabe's WBIC: E_w^\beta[nL_n(w)] = nL_n(w_0) + \lambda/\beta + U_n \sqrt(\lambda/\beta)
def lambda_thm4(betas, args, kwargs):

    RLCT_estimates_GLS = np.empty(0)
    RLCT_estimates_OLS = np.empty(0)

    for mc in range(0, args.MCs):

        print('Starting MC {}'.format(mc))
        # draw new training-testing split

        train_loader, valid_loader, test_loader, input_dim, output_dim, loss, true_RLCT = get_dataset_by_id(args, kwargs)

        temperedNLL_perMC_perBeta = np.empty(0)
        for beta in betas:
            temp, _ = approxinf_nll(train_loader, valid_loader, test_loader, input_dim, output_dim, args, mc, beta, betas)
            temperedNLL_perMC_perBeta = np.append(temperedNLL_perMC_perBeta, temp)

        # least squares fit for lambda
        ols, gls = lsfit_lambda(temperedNLL_perMC_perBeta, betas)
        if gls != None:
            RLCT_estimates_GLS = np.append(RLCT_estimates_GLS, gls)
            RLCT_estimates_OLS = np.append(RLCT_estimates_OLS, ols)

            plt.scatter(1 / betas, temperedNLL_perMC_perBeta)
            plt.title("Thm 4, one MC realisation: hat lambda = {:.2f}, true lambda = {:.2f}".format(gls, args.w_dim/2))
            plt.xlabel("1/beta")
            plt.ylabel("implicit VI estimate of E^beta_w [nL_n(w)]")
            #plt.show()

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
            temp, _ = approxinf_nll(train_loader,
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
            _, nlls = approxinf_nll(train_loader, test_loader, input_dim, output_dim, args, beta1)

            lambda_beta1 = (nlls.mean() - (nlls * np.exp(-(beta2 - beta1) * nlls)).mean() / (np.exp(-(beta2 - beta1) * nlls)).mean()) / (1 / beta1 - 1 / beta2)
            lambdas_beta1 = np.append(lambdas_beta1, lambda_beta1)
            RLCT_estimates = np.append(RLCT_estimates, lambdas_beta1.mean())


        print('MC: {} RLCT estimate: {:.2f}'.format(mc, lambdas_beta1.mean()))

    return RLCT_estimates


def main():
    if not os.path.exists('./result'):
        os.makedirs('./result')

    random.seed()

    # Training settings
    parser = argparse.ArgumentParser(description='RLCT Implicit Variational Inference')
    parser.add_argument('--taskid', type=int, default=0,
                        help='taskid from sbatch')
    parser.add_argument('--dataset', type=str, default='lr_synthetic',
                        help='dataset name from dataset_factory.py (default: )',
                        choices=['iris-binary', 'breastcancer-binary', 'MNIST-binary', 'MNIST','lr_synthetic', '3layertanh_synthetic', 'reducedrank_synthetic'])

    parser.add_argument('--syntheticsamplesize', type=int, default=60000,
                        help='sample size of synthetic dataset')

    parser.add_argument('--VItype', type=str, default='implicit',
                        help='type of variaitonal inference',
                        choices=['explicit','implicit'])

    parser.add_argument('--network', type=str, default='logistic',
                        help='name of network in models.py (default: logistic)',
                        choices=['FFrelu','CNN','logistic', 'Tanh', 'ReducedRankRegression'])

    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')

    parser.add_argument('--batchsize', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')

    parser.add_argument('--betasbegin', type=float, default=0.1,
                        help='where beta range should begin')

    parser.add_argument('--betasend', type=float, default=2,
                        help='where beta range should end')

    parser.add_argument('--betalogscale', type=str, default='true',
                        help='true if beta should be on 1/log n scale (default: true)',
                        choices=['true','false'])

    parser.add_argument('--n_hidden_D', type=int, default=256,
                        help='number of hidden units in discriminator D')

    parser.add_argument('--num_hidden_layers_D', type = int, default=2,
                        help='number of hidden layers in discriminatror D')

    parser.add_argument('--n_hidden_G', type=int, default=256,
                        help='number of hidden units in generator G')

    parser.add_argument('--num_hidden_layers_G', type = int, default = 2,
                        help = 'number of hidden layers in generator G')

    parser.add_argument('--lambda_asymptotic', type=str, default='thm4',
                        help='which asymptotic characterisation of lambda to use',
                        choices=['thm4', 'thm4_average', 'cor3'])

    parser.add_argument('--pretrainDepochs', type=int,  default=2,
                        help='number of epochs to pretrain discriminator')
    
    parser.add_argument('--trainDepochs', type=int, default=2,
                        help='number of epochs to train discriminator for each minibatch update of generator')
    
    parser.add_argument('--dpower', type=float, default=2/5,
                        help='set dimension of model to n^dpower')
    
    # as high as possible
    # parser.add_argument('--epsilon_mc', type=int, default=10,
    #                     help='number of draws for estimating E_\epsilon')
    parser.add_argument('--numbetas', type=int,  default=20,
                        help='how many betas should be swept between betasbegin and betasend')

    parser.add_argument('--MCs', type=int, default=100,
                        help='number of times to split into train-test')

    parser.add_argument('--R', type=int, default=100,
                        help='number of MC draws from approximate posterior (default:100)')

    parser.add_argument('--wandb_on', action="store_true",
                        help='use wandb to log experiment')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--lr_primal', type=float,  default=1e-4, metavar='LR',
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
                        help='prior used on model parameters (default: gaussian)',
                        choices=['gaussian', 'mixtgauss', 'conjugate', 'conjugate_known_mean'])
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=62364)

    args = parser.parse_args()
    print(vars(args))

    # TODO: maybe should switch to Tensorboard, wandb does not seem to have many users
    if args.wandb_on:
        import wandb
        wandb.init(project="RLCT", entity="unimelb_rlct")
        wandb.config.update(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("args.cuda is " + str(args.cuda))

    # Daniel
    # torch.manual_seed(args.seed)
    # if args.cuda:
    #    torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # set true network weights for synthetic dataset
    if args.dataset == 'lr_synthetic':
        input_dim = int(np.power(args.syntheticsamplesize*0.7, args.dpower))
        args.w_0 = torch.randn(input_dim,1)
        args.b = torch.randn(1)
    elif args.dataset == '3layertanh_synthetic':
        H = int(np.power(args.syntheticsamplesize*0.7, args.dpower)*0.5) #number of hidden unit
        args.H = H
        args.a_params = torch.zeros([1, H], dtype=torch.float32)
        args.b_params = torch.zeros([H, 1], dtype=torch.float32)

    elif args.dataset == 'reducedrank_synthetic':
        #suppose input_dimension=output_dimension + 3, H = output_dimension, H is number of hidden nuit
        #solve the equation (input_dimension + output_dimension)*H = np.power(args.syntheticsamplesize, args.dpower) to get output_dimension, then input_dimension, and H
        output_dim = int((-3 + math.sqrt(9 + 4*2*np.power(args.syntheticsamplesize*0.7, args.dpower)))/4)
        H = output_dim
        input_dim = output_dim + 3
        args.H = H
        args.a_params = torch.cat((torch.eye(H),torch.ones([input_dim-H, H],dtype=torch.float32)), 0)
        args.b_params = torch.eye(output_dim)
        #in this case, the rank r for args.a_params * args.b_params is H, output_dim + H < input_dim + r is satisfied

    # draw a training-testing split just to get some necessary parameters
    train_loader, valid_loader, test_loader, input_dim, output_dim, loss, true_RLCT = get_dataset_by_id(args, kwargs)
    args.n = len(train_loader.dataset)
    args.input_dim = input_dim
    args.output_dim = output_dim
    args.loss = loss

    # retrieve model
    model, w_dim = retrieve_model(args, input_dim, output_dim)

    args.model = model
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
            "true_RLCT": true_RLCT,
            "RLCT_estimates_OLS": RLCT_estimates_OLS,
            "RLCT_estimates_GLS": RLCT_estimates_GLS,
            "mean_of_RLCT_estimates_OLS": RLCT_estimates_OLS.mean(),
            "std_of_RLCT_estimates_OLS": RLCT_estimates_OLS.std(),
            "mean_of_RLCT_estimates_GLS": RLCT_estimates_GLS.mean(),
            "std_of_RLCT_estimates_GLS": RLCT_estimates_GLS.std(),
            "d_on_2": w_dim/2
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

    pd.DataFrame.from_dict(results).to_csv('./result/RLCTestimate_taskid%s.csv'%(args.taskid))

    print(results)
    if args.wandb_on:
        wandb.log(results)
        f = open("results.pkl", "wb")
        pickle.dump(results, f)
        f.close()
        wandb.save("results.pkl")

if __name__ == "__main__":
    main()


