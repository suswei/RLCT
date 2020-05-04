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
from joblib import Parallel, delayed
import random
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import copy
import pickle
import itertools
import math
import pyvarinf
import models
from dataset_factory import get_dataset_by_id
from RLCT_helper import *
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from random import randint
# from lambda_asymptotics import *


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
def train_implicitVI(train_loader,valid_loader,args, mc, beta_index, saveimgpath):

    # instantiate generator and discriminator
    G_initial = Generator(args.epsilon_dim, args.w_dim, args.n_hidden_G, args.num_hidden_layers_G)  # G = Generator(args.epsilon_dim, w_dim).to(args.cuda)
    D_initial = Discriminator(args.w_dim, args.n_hidden_D, args.num_hidden_layers_D)  # D = Discriminator(w_dim).to(args.cuda)
    G = copy.deepcopy(G_initial)
    D = copy.deepcopy(D_initial)

    # optimizers
    opt_primal = optim.Adam(
        G.parameters(),
        lr=args.lr_primal)
    opt_dual = optim.Adam(
        D.parameters(),
        lr=args.lr_dual)
    scheduler_G = ReduceLROnPlateau(opt_primal, mode='min', factor=0.1, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=10, verbose=True)

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

    train_loss_epoch, valid_loss_epoch, train_reconstr_err_epoch, valid_reconstr_err_epoch, D_err_epoch = [], [], [], [], []
    reconstr_err_minibatch, D_err_minibatch, primal_loss_minibatch = [], [], []

    # train discriminator and generator together
    for epoch in range(args.epochs):

        D.train()
        G.train()

        if args.dataset == 'logistic_synthetic':
            correct = 0
        elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
            training_sum_se = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            # opt discriminator more than generator
            for discriminator_epoch in range(args.trainDepochs):

                w_sampled_from_prior = randn((args.epsilon_mc, args.w_dim), args.cuda)  # TODO: add more options for prior besides hardcoding Gaussian prior
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

                if args.dataset == 'logistic_synthetic':

                    if args.bias:
                        A = w_sampled_from_G[i, 0:(args.w_dim - 1)]
                        b = w_sampled_from_G[i, args.w_dim - 1]
                        output = torch.mm(data, A.reshape(args.w_dim - 1, 1)) + b
                    else:
                        A = w_sampled_from_G[i, 0:(args.w_dim)]
                        output = torch.mm(data, A.reshape(args.w_dim, 1))
                    output_cat_zero = torch.cat((output, torch.zeros(data.shape[0], 1)), 1)
                    logsoftmax_output = F.log_softmax(output_cat_zero, dim=1)
                    # input to nll_loss should be log-probabilities of each class. input has to be a Tensor of size (minibatch, C)
                    reconstr_err += args.loss_criterion(logsoftmax_output, target)

                elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:

                    a_params = w_sampled_from_G[i, 0:(args.input_dim * args.H)].reshape(args.H, args.input_dim)
                    b_params = w_sampled_from_G[i, (args.input_dim * args.H):].reshape(args.output_dim, args.H)
                    if args.dataset == 'tanh_synthetic':
                        output = torch.matmul(torch.tanh(torch.matmul(data, torch.transpose(a_params, 0, 1))), torch.transpose(b_params, 0, 1))
                    else:
                        output = torch.matmul(torch.matmul(data, torch.transpose(a_params, 0, 1)), torch.transpose(b_params, 0, 1))
                    reconstr_err += args.loss_criterion(output, target) #reduction is set to be 'mean' by default

            if args.dataset in ('reducedrank_synthetic','tanh_synthetic'):
                reconstr_err_component = reconstr_err / (2*args.epsilon_mc)
            else:
                reconstr_err_component = reconstr_err / args.epsilon_mc

            discriminator_err_component = torch.mean(D(w_sampled_from_G)) / (args.betas[beta_index] * args.n)
            loss_primal = reconstr_err_component + discriminator_err_component
            loss_primal.backward(retain_graph=True)
            opt_primal.step()
            G.zero_grad()
            D.zero_grad()

            reconstr_err_minibatch.append(reconstr_err_component.item())
            D_err_minibatch.append(discriminator_err_component.item())
            primal_loss_minibatch.append(loss_primal.item())

            # minibatch logging on args.log_interval
            if args.dataset == 'logistic_synthetic':
                pred = logsoftmax_output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data).cpu().sum()
            elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
                training_sum_se += args.loss_criterion(output, target).detach().cpu().numpy()*len(target)
            if batch_idx % args.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss primal: {:.6f}\tLoss dual: {:.6f}'.format(
                        epoch, batch_idx * len(data), args.n,
                               100. * batch_idx / len(train_loader), loss_primal.data.item(), loss_dual.data.item()))

        # every epoch, log the following
        if args.dataset == 'logistic_synthetic':
            print('\nTrain set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, args.n, 100. * correct / args.n))
        elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
            print('\nTrain set: MSE: {} \n'.format(training_sum_se/args.n))

        with torch.no_grad(): #to save memory, no intermediate activations used for gradient calculation is stored.
            D.eval()
            G.eval()
            #valid loss is calculated for all monte carlo as it is used for scheduler_G
            valid_loss_minibatch = []
            for valid_batch_id, (valid_data, valid_target) in enumerate(valid_loader):
                valid_data, valid_target = load_minibatch(args, valid_data, valid_target)
                if args.dataset == 'tanh_synthetic':
                   valid_output = torch.matmul(torch.tanh(torch.matmul(valid_data, torch.transpose(a_params, 0, 1))), torch.transpose(b_params, 0, 1))
                   valid_loss_minibatch.append(args.loss_criterion(valid_output, valid_target).item() * 0.5)
                elif args.dataset =='logistic_synthetic':
                    if args.bias:
                        output = torch.mm(valid_data, A.reshape(args.w_dim - 1, 1)) + b
                    else:
                        output = torch.mm(valid_data, A.reshape(args.w_dim, 1))
                    output_cat_zero = torch.cat((output, torch.zeros(valid_data.shape[0], 1)), 1)
                    valid_output = F.log_softmax(output_cat_zero, dim=1)
                    valid_loss_minibatch.append(args.loss_criterion(valid_output, valid_target).item())
                elif args.dataset == 'reducedrank_synthetic':
                   valid_output = torch.matmul(torch.matmul(valid_data, torch.transpose(a_params, 0, 1)), torch.transpose(b_params, 0, 1))
                   valid_loss_minibatch.append(args.loss_criterion(valid_output, valid_target).item()*0.5)
            valid_loss_one = np.average(valid_loss_minibatch) + torch.mean(D(w_sampled_from_G)) / (args.betas[beta_index]  * args.n)
            scheduler_G.step(valid_loss_one)
            valid_loss_epoch.append(valid_loss_one)
            valid_reconstr_err_epoch.append(np.average(valid_loss_minibatch))
            D_err_epoch.append(torch.mean(D(w_sampled_from_G)) / (args.betas[beta_index]  * args.n))

            train_loss_minibatch2 = []
            for train_batch_id, (train_data, train_target) in enumerate(train_loader):
                train_data, train_target = load_minibatch(args, train_data, train_target)
                if args.dataset == 'tanh_synthetic':
                    train_output = torch.matmul(torch.tanh(torch.matmul(train_data, torch.transpose(a_params, 0, 1))), torch.transpose(b_params, 0, 1))
                    train_loss_minibatch2.append(args.loss_criterion(train_output, train_target).item()*0.5)
                elif args.dataset == 'logistic_synthetic':
                    if args.bias:
                        output = torch.mm(train_data, A.reshape(args.w_dim - 1, 1)) + b
                    else:
                        output = torch.mm(train_data, A.reshape(args.w_dim, 1))

                    output_cat_zero = torch.cat((output, torch.zeros(train_data.shape[0], 1)), 1)
                    train_output = F.log_softmax(output_cat_zero, dim=1)
                    train_loss_minibatch2.append(args.loss_criterion(train_output, train_target).item())
                elif args.dataset == 'reducedrank_synthetic':
                    train_output = torch.matmul(torch.matmul(train_data, torch.transpose(a_params, 0, 1)), torch.transpose(b_params, 0, 1))
                    train_loss_minibatch2.append(args.loss_criterion(train_output, train_target).item()*0.5)
            train_loss_epoch.append(np.average(train_loss_minibatch2) + torch.mean(D(w_sampled_from_G)) / (args.betas[beta_index]  * args.n))
            train_reconstr_err_epoch.append(np.average(train_loss_minibatch2))

        # early_stopping(valid_loss_one, G)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    plt.figure(figsize=(10, 7))
    plt.plot(list(range(0, len(train_loss_epoch))), train_loss_epoch,
             list(range(0, len(valid_loss_epoch))), valid_loss_epoch,
             list(range(0, len(train_reconstr_err_epoch))), train_reconstr_err_epoch,
             list(range(0, len(valid_reconstr_err_epoch))), valid_reconstr_err_epoch,
             list(range(0, len(D_err_epoch))), D_err_epoch)
    plt.legend(('primal loss (train)', 'primal loss (validation)', 'reconstr err component (train)', 'reconstr err component (valid)', 'discriminator err component'), loc='center right', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    plt.title('beta = {}'.format(args.betas[beta_index]), fontsize=18)
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(list(range(0, len(reconstr_err_minibatch)))[20:], reconstr_err_minibatch[20:],
             list(range(0, len(D_err_minibatch)))[20:], D_err_minibatch[20:],
             list(range(0, len(primal_loss_minibatch)))[20:], primal_loss_minibatch[20:]
    )

    plt.legend(('reconstr err component', 'discriminator err component','primal loss'), loc='upper right', fontsize=16)
    plt.xlabel('epochs*batches (minibatches)', fontsize=16)
    plt.title('training_set, beta = {}'.format(args.betas[beta_index]), fontsize=18)
    plt.savefig('./{}/reconsterr_derr_minibatch_betaind{}.png'.format(saveimgpath, beta_index))
    plt.close()

    return G


def train_explicitVI(train_loader,valid_loader, args, mc, beta_index, verbose, saveimgpath):


    # retrieve model
    model, _ = retrieve_model(args)

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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_loss_epoch, valid_loss_epoch, train_reconstr_err_epoch, valid_reconstr_err_epoch, loss_prior_epoch = [], [], [], [], []
    reconstr_err_minibatch, loss_prior_minibatch, train_loss_minibatch= [], [], []
    # train var_model
    for epoch in range(1, args.epochs + 1):

        var_model.train()

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = load_minibatch(args, data, target)

            optimizer.zero_grad()
            # var_model draw a sample of the network parameter and then applies the network with the sampled weights
            output = var_model(data)
            # when applying var_model, can see var_model.dico epsilon is always zero, this means E_variational L_n(w) is estimated at L_n(mean(dataset))
            # prior_loss takes into account the var_model.dico['linear']['weight'].rho

            loss_prior = var_model.prior_loss() / (args.betas[beta_index]*args.n)
            
            if args.dataset == 'logistic_synthetic':
                # reconstr_err = args.loss_criterion(output, target)
                BCE_loss = torch.nn.BCELoss()
                reconstr_err = BCE_loss(1-output, target.float())
            elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
                reconstr_err = args.loss_criterion(output, target)*0.5
            
            loss = reconstr_err + loss_prior  # this is the ELBO
            loss.backward()
            optimizer.step()

            reconstr_err_minibatch.append(reconstr_err.item())
            loss_prior_minibatch.append(loss_prior.item())
            train_loss_minibatch.append(loss.item())

            if verbose:
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss error: {:.6f}\tLoss weights: {:.6f}'.format(
                        epoch, batch_idx * len(data), args.n,
                        100. * batch_idx / len(train_loader), loss.data.item(), reconstr_err.data.item(), loss_prior.data.item()))

        with torch.no_grad():  # to save memory, no intermediate activations used for gradient calculation is stored.
            var_model.eval()
            # valid loss is calculated for all monte carlo as it is used for scheduler_G
            valid_loss_minibatch = []
            for valid_batch_id, (valid_data, valid_target) in enumerate(valid_loader):
                valid_data, valid_target = load_minibatch(args, valid_data, valid_target)
                valid_output = var_model(valid_data)
                if args.dataset == 'logistic_synthetic':
                    # valid_loss_minibatch.append(args.loss_criterion(valid_output, valid_target).item())
                    BCE_loss = torch.nn.BCELoss()
                    valid_loss_minibatch.append(BCE_loss(1-valid_output, valid_target.float()).item())
                elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
                    valid_loss_minibatch.append(args.loss_criterion(valid_output, valid_target).item()*0.5)
            valid_loss_one = np.average(valid_loss_minibatch) + var_model.prior_loss() / (args.betas[beta_index] * args.n)
            scheduler.step(valid_loss_one)
            valid_loss_epoch.append(valid_loss_one)
            valid_reconstr_err_epoch.append(np.average(valid_loss_minibatch))
            loss_prior_epoch.append(var_model.prior_loss() / (args.betas[beta_index] * args.n))

            train_loss_minibatch2 = []
            for train_batch_id, (train_data, train_target) in enumerate(train_loader):
                train_data, train_target = load_minibatch(args, train_data, train_target)
                train_output = var_model(train_data)
                if args.dataset == 'logistic_synthetic':
                    # train_loss_minibatch2.append(args.loss_criterion(train_output, train_target).item())
                    BCE_loss = torch.nn.BCELoss()
                    train_loss_minibatch2.append(BCE_loss(1 - train_output, train_target.float()).item())
                elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
                    train_loss_minibatch2.append(args.loss_criterion(train_output, train_target).item()*0.5)
            train_loss_epoch.append(np.average(train_loss_minibatch2)+var_model.prior_loss() / (args.betas[beta_index] * args.n))
            train_reconstr_err_epoch.append(np.average(train_loss_minibatch2))

        # early_stopping(valid_loss_one, var_model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    plt.figure(figsize=(10, 7))
    plt.plot(list(range(0, len(train_loss_epoch))), train_loss_epoch,
             list(range(0, len(valid_loss_epoch))), valid_loss_epoch,
             list(range(0, len(train_reconstr_err_epoch))), train_reconstr_err_epoch,
             list(range(0, len(valid_reconstr_err_epoch))), valid_reconstr_err_epoch,
             list(range(0, len(loss_prior_epoch))), loss_prior_epoch)
    plt.legend(('loss (train)', 'loss (validation)', 'reconstr err component (train)',
                'reconstr err component (valid)', 'loss prior component'), loc='center right', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    plt.title('beta = {}'.format(args.betas[beta_index]), fontsize=18)
    plt.savefig('./{}/primal_loss_betaind{}.png'.format(saveimgpath, beta_index))
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(list(range(0, len(reconstr_err_minibatch)))[20:], reconstr_err_minibatch[20:],
             list(range(0, len(loss_prior_minibatch)))[20:], loss_prior_minibatch[20:],
             list(range(0, len(train_loss_minibatch)))[20:], train_loss_minibatch[20:])

    plt.legend(('reconstr err component', 'loss prior component', 'loss'), loc='upper right',fontsize=16)
    plt.xlabel('epochs*batches (minibatches)', fontsize=16)
    plt.title('training_set, beta = {}'.format(args.betas[beta_index]), fontsize=18)
    plt.savefig('./{}/reconsterr_derr_minibatch_betaind{}.png'.format(saveimgpath, beta_index))
    plt.close()

    return var_model


def sample_weights_from_explicitVI(sample, args):

    sampled_weight = torch.empty((0, args.w_dim))

    # TODO: no for loop is necessary actually
    for draw_index in range(100):

        if args.dataset == 'logistic_synthetic':

            temp = sample.var_model.dico['linear']['weight']
            weight = temp.mean + (1 + temp.rho.exp()).log() * torch.randn(temp.mean.shape)
            if args.bias:
                temp = sample.var_model.dico['linear']['bias']
                bias = temp.mean + (1 + temp.rho.exp()).log() * torch.randn(temp.mean.shape)
                weight_bias = torch.cat((weight[0, :], bias[0].unsqueeze(dim=0)), 0).unsqueeze(dim=0)
            else:
                weight_bias = weight[0, :].unsqueeze(dim=0)

            # weight_mean_rho_eps = list(list(sample.var_model.dico.values())[0].values())[0]
            # bias_mean_rho_eps = list(list(sample.var_model.dico.values())[0].values())[1]
            # weight = (weight_mean_rho_eps.mean + (1 + weight_mean_rho_eps.rho.exp()).log() * weight_mean_rho_eps.eps)[0,:].reshape(1, weight_mean_rho_eps.mean.shape[1])  # there are two probability output for each category, we only need the parameters for one category
            # bias = (bias_mean_rho_eps.mean + (1 + bias_mean_rho_eps.rho.exp()).log() * bias_mean_rho_eps.eps)[0].reshape(1, 1)
            # weight_bias = torch.cat((weight, bias), 1)

            sampled_weight = torch.cat((sampled_weight, weight_bias), 0)

        # TODO: Hui's code probably doesn't work either for the same reasons it didn't work for lr_synthetic
        elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
            layer1_weight_mean_rho_eps = list(list(sample.var_model.dico.values())[0].values())[0]
            layer2_weight_mean_rho_eps = list(list(sample.var_model.dico.values())[1].values())[0]

            layer1_weight = layer1_weight_mean_rho_eps.mean + (
                        1 + layer1_weight_mean_rho_eps.rho.exp()).log() * layer1_weight_mean_rho_eps.eps  # H * input_dim
            layer2_weight = layer2_weight_mean_rho_eps.mean + (
                        1 + layer2_weight_mean_rho_eps.rho.exp()).log() * layer2_weight_mean_rho_eps.eps  # output_dim * H

            one_weight = torch.cat((layer1_weight.reshape(1, (layer1_weight.shape[0] * layer1_weight.shape[1])),
                                    layer2_weight.reshape(1, (layer2_weight.shape[0] * layer2_weight.shape[1]))), 1)
            sampled_weight = torch.cat((sampled_weight, one_weight), 0)

    return sampled_weight


# TODO: approxinf_nll_implicit and approxinf_nll_explicit should eventually be merged.
# Draw w^* from generator G and evaluate nL_n(w^*) on train_loader
def approxinf_nll_implicit(r, train_loader, G, args):

    G.eval()
    with torch.no_grad():

        eps = randn((1, args.epsilon_dim), args.cuda)
        w_sampled_from_G = G(eps)
        w_dim = w_sampled_from_G.shape[1]
        if args.dataset == 'logistic_synthetic':
            if args.bias:
                A = w_sampled_from_G[0, 0:(w_dim - 1)]
                b = w_sampled_from_G[0, w_dim - 1]
            else:
                A = w_sampled_from_G[0, 0:(w_dim)]
                b = 0.0
        elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
            a_params = w_sampled_from_G[0, 0:(args.input_dim * args.H)].reshape(args.H, args.input_dim)
            b_params = w_sampled_from_G[0, (args.input_dim * args.H):].reshape(args.output_dim, args.H)

        nll = np.empty(0)
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = load_minibatch(args, data, target)
            if args.dataset == 'logistic_synthetic':
                if args.bias:
                    output = torch.mm(data, A.reshape(w_dim - 1, 1)) + b
                else:
                    output = torch.mm(data, A.reshape(w_dim, 1))

                output_cat_zero = torch.cat((output, torch.zeros(data.shape[0], 1)), 1)
                output = F.log_softmax(output_cat_zero, dim=1)
                # input to nll_loss should be log-probabilities of each class. input has to be a Tensor of size either (minibatch, C)
            elif args.dataset == 'tanh_synthetic':
                output = torch.matmul(torch.tanh(torch.matmul(data, torch.transpose(a_params, 0, 1))), torch.transpose(b_params, 0, 1))
            elif args.dataset == 'reducedrank_synthetic':
                output = torch.matmul(torch.matmul(data, torch.transpose(a_params, 0, 1)), torch.transpose(b_params, 0, 1))

            nll_new = args.loss_criterion(output, target)*len(target) #get sum loss
            nll = np.append(nll, np.array(nll_new.detach().cpu().numpy()))

    if args.dataset == 'logistic_synthetic':
        return nll.sum()
    elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
        return target.shape[1]/2*np.log(2*np.pi)+nll.sum()/2
    else:
        print('misspelling in dataset name!')


# w^* is drawn by calling sample.draw(), this function evaluates nL_n(w^*) on train_loader
def approxinf_nll_explicit(r, train_loader, sample, args):

    if args.dataset in ['tanh_synthetic','reducedrank_synthetic']:
        MSEloss = nn.MSELoss(reduction='sum')
    loss = np.empty(0)


    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = load_minibatch(args, data, target)
        sample.draw()
        output = sample(data)
        # can check at this point sample.var_model.dico to see the epsilon parameters change due to sample.draw
        if args.dataset == 'logistic_synthetic':
            loss = np.append(loss, np.array(F.nll_loss(output, target, reduction="sum").detach().cpu().numpy()))
        elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
            loss = np.append(loss, np.array(MSEloss(output, target).detach().cpu().numpy()))

    if args.dataset == 'logistic_synthetic':
        return loss.sum()
    elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
        return target.shape[1]/2*np.log(2*np.pi)+loss.sum()/2


# TODO: At the moment should only use with lr_synthetic and two total parameters (no bias!)
def posterior_viz(train_loader, sampled_weights,args,beta_index,saveimgpath):

    wmin = -3
    wmax = 3
    wspace = 50
    wrange = np.linspace(wmin, wmax, wspace)
    w = np.repeat(wrange[:, None], wspace, axis=1)
    w = np.concatenate([[w.flatten()], [w.T.flatten()]])

    logprior = -(w ** 2).sum(axis=0) / 2
    plt.contourf(wrange, wrange, logprior.reshape(wspace, wspace), cmap='gray')
    plt.axis('square')
    plt.xlim([wmin, wmax])
    plt.ylim([wmin, wmax])
    plt.title('log prior (unnormalised): -1/2 w^Tw')
    plt.savefig('./{}/prior_betaind{}.png'.format(saveimgpath,beta_index))
    plt.close()

    logpost = torch.zeros(wspace * wspace)
    for i in range(wspace * wspace):
        A = torch.Tensor(w[:, i])
        b = args.b
        data = train_loader.dataset[:][0]
        target = train_loader.dataset[:][1]
        output = torch.mm(data, A.reshape(args.w_dim, 1)) + b
        output_cat_zero = torch.cat((output, torch.zeros(data.shape[0], 1)), 1)
        logsoftmax_output = F.log_softmax(output_cat_zero, dim=1)
        # input to nll_loss should be log-probabilities of each class. input has to be a Tensor of size (minibatch, C)
        logpost[i] = -args.betas[beta_index]* args.loss_criterion(logsoftmax_output, target)*len(target)

    logpost = logpost.detach().numpy() + logprior
    plt.contourf(wrange, wrange, np.exp(logpost.reshape(wspace, wspace).T), cmap='gray')
    plt.axis('square')
    sampled_weights = sampled_weights.detach().numpy()
    plt.plot(sampled_weights[:, 0], sampled_weights[:, 1], '.g')
    plt.xlim([wmin, wmax])
    plt.ylim([wmin, wmax])
    plt.title('true (unnormalised) posterior at inverse temp {}'.format(args.betas[beta_index]))
    plt.savefig('./{}/posteior_betaind{}.png'.format(saveimgpath,beta_index))
    plt.show()
    plt.close()


def tsne_viz(sampled_weights,true_weight,args,beta_index,saveimgpath):

    sampled_true_w = torch.cat((sampled_weights, true_weight), 0)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(sampled_true_w.detach().numpy())
    tsne_results = pd.DataFrame(tsne_results, columns=['dim%s' % (dim_index) for dim_index in [1, 2]])
    tsne_results = pd.concat(
        [tsne_results, pd.DataFrame.from_dict({'sampled_true': ['sampled'] * (tsne_results.shape[0] - 1) + ['true']})],
        axis=1)
    plt.figure(figsize=(16, 10))
    ax = sns.scatterplot(x="dim1", y="dim2", hue="sampled_true", data=tsne_results)
    plt.title('tsne view: sampled w: beta = {}'.format(args.betas[beta_index]), fontsize=20)
    plt.savefig('./{}/tsne_betaind{}.png'.format(saveimgpath, beta_index))
    plt.close()


# Approximate inference estimate of E_w^\beta [nL_n(w)]:  1/R \sum_{r=1}^R nL_n(w_r^*)
# Also return approximate inference estimate of var_w^\beta [nL_n(w)]: 1/R \sum_{r=1}^R [nL_n(w_r^*)]^2 - [ 1/R \sum_{r=1}^R nL_n(w_r^*)]^2
def approxinf_nll(train_loader,valid_loader,args, mc, beta_index, saveimgpath):

    args.epsilon_dim = args.w_dim
    args.epsilon_mc = args.batchsize  # TODO: overwriting args parser input

    if args.dataset == 'logistic_synthetic':
        true_weight = torch.cat((args.w_0.reshape(1, args.input_dim), args.b.reshape(1, 1)), 1)
    elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
        true_weight = torch.cat((args.a_params.reshape(1, (args.a_params.shape[0] * args.a_params.shape[1])), args.b_params.reshape(1, (args.b_params.shape[0] * args.b_params.shape[1]))), 1)

    if args.VItype == 'implicit':

        G = train_implicitVI(train_loader,valid_loader,args, mc, beta_index, saveimgpath)

        # visualize generator G
        with torch.no_grad():
            eps = torch.randn(100, args.epsilon_dim)
            w_sampled_from_G = G(eps)

            if args.tsne_viz:
                tsne_viz(w_sampled_from_G, true_weight, args, beta_index, saveimgpath)

            if args.posterior_viz:
                posterior_viz(train_loader, w_sampled_from_G, args, beta_index, saveimgpath)

        my_list = range(args.R)
        num_cores = 1  # multiprocessing.cpu_count()
        # approxinf_nlls returns array [nL_n(w_1^*),\ldots, nL_n(w_R^*)] where w^* is drawn from generator G
        approxinf_nlls = Parallel(n_jobs=num_cores, verbose=0)(
            delayed(approxinf_nll_implicit)(i, train_loader,G, args) for i in my_list)

    elif args.VItype == 'explicit':

        var_model = train_explicitVI(train_loader, valid_loader, args, mc, beta_index, True, saveimgpath)

        with torch.no_grad():

            # form sample object from variational distribution r
            sample = pyvarinf.Sample(var_model=var_model)

            if args.tsne_viz or args.posterior_viz:
                sampled_weights = sample_weights_from_explicitVI(sample, args)

            if args.tsne_viz:
                tsne_viz(sampled_weights, true_weight, args, beta_index, saveimgpath)

            if args.posterior_viz:
                posterior_viz(train_loader, sampled_weights, args, beta_index, saveimgpath)

        # draws R samples {w_1,\ldots,w_R} from r_\theta^\beta (var_model) and returns \frac{1}{R} \sum_{i=1}^R [nL_n(w_i}]
        my_list = range(args.R)
        num_cores = 1  # multiprocessing.cpu_count()
        approxinf_nlls = Parallel(n_jobs=num_cores, verbose=0)(
            delayed(approxinf_nll_explicit)(i, train_loader,sample, args) for i in my_list)

    approxinf_nlls_array = np.asarray(approxinf_nlls)
    mean_nlls = approxinf_nlls_array.mean()
    var_nlls = (approxinf_nlls_array**2).mean() - (mean_nlls)**2
    return mean_nlls, var_nlls, approxinf_nlls_array

# Thm 4 of Watanabe's WBIC: E_w^\beta[nL_n(w)] = nL_n(w_0) + \lambda/\beta + U_n \sqrt(\lambda/\beta)
def lambda_thm4(args, kwargs):

    RLCT_estimates_OLS = np.empty(0)
    RLCT_estimates_robust = np.empty(0)

    for mc in range(0, args.MCs):

        path = './{}_sanity_check/taskid{}/img/mc{}'.format(args.VItype, args.taskid, mc)
        if not os.path.exists(path):
            os.makedirs(path)

        print('Starting MC {}'.format(mc))
        # draw new training-testing split

        train_loader, valid_loader, test_loader, input_dim, output_dim, loss, true_RLCT = get_dataset_by_id(args, kwargs)

        temperedNLL_perMC_perBeta = np.empty(0)
        for beta_index in range(args.betas.shape[0]):
            temp, _, _ = approxinf_nll(train_loader, valid_loader, args, mc, beta_index, path)
            temperedNLL_perMC_perBeta = np.append(temperedNLL_perMC_perBeta, temp)

        # least squares fit for lambda
        robust, ols = lsfit_lambda(temperedNLL_perMC_perBeta, args, path)

        RLCT_estimates_robust = np.append(RLCT_estimates_robust, robust)
        RLCT_estimates_OLS = np.append(RLCT_estimates_OLS, ols)

        print("RLCT robust: {}".format(RLCT_estimates_robust))
        print("RLCT OLS: {}".format(RLCT_estimates_OLS))

        if args.wandb_on:
            import wandb
            wandb.run.summary["running RLCT robust"] = RLCT_estimates_robust
            wandb.run.summary["running RLCT OLS"] = RLCT_estimates_OLS

        print('Finishing MC {}'.format(mc))

    # return array of RLCT estimates, length args.MCs
    return RLCT_estimates_robust, RLCT_estimates_OLS


# apply E_{D_n} to Theorem 4 of Watanabe's WBIC: E_{D_n} E_w^\beta[nL_n(w)] = E_{D_n} nL_n(w_0) + \lambda/\beta
def lambda_thm4average(args, kwargs):

    temperedNLL_perBeta = np.empty(0)

    for beta_index in range(args.betas.shape[0]):

        beta = args.betas[beta_index]
        print('Starting beta {}'.format(beta))

        temperedNLL_perMC_perBeta = np.empty(0)

        for mc in range(0, args.MCs):

            path = './{}_sanity_check/taskid{}/img/beta{}/mc{}'.format(args.VItype, args.taskid, beta, mc)
            if not os.path.exists(path):
                os.makedirs(path)

            # draw new training-testing split
            train_loader, valid_loader, test_loader, input_dim, output_dim, loss, true_RLCT = get_dataset_by_id(args, kwargs)
            temp,_,_ = approxinf_nll(train_loader, valid_loader, args, mc, beta_index, path)
            temperedNLL_perMC_perBeta = np.append(temperedNLL_perMC_perBeta, temp)

        temperedNLL_perBeta = np.append(temperedNLL_perBeta, temperedNLL_perMC_perBeta.mean())

        print('Finishing beta {}'.format(beta))

    path = './{}_sanity_check/taskid{}/img/'.format(args.VItype, args.taskid)
    if not os.path.exists(path):
        os.makedirs(path)

    RLCT_estimate_robust, RLCT_estimate_OLS = lsfit_lambda(temperedNLL_perBeta, args, path)

    # each RLCT estimate is one elment array
    return RLCT_estimate_robust, RLCT_estimate_OLS


def lambda_cor3(args, kwargs):

    RLCT_estimates = np.empty(0)

    for mc in range(0, args.MCs):

        path = './{}_sanity_check/taskid{}/img/mc{}'.format(args.VItype, args.taskid, mc)
        if not os.path.exists(path):
            os.makedirs(path)

        # draw new training-testing split
        train_loader, valid_loader, test_loader, input_dim, output_dim, loss, true_RLCT = get_dataset_by_id(args, kwargs)

        lambdas_beta1 = np.empty(0)
        for beta_index in range(args.betas.shape[0]):

            beta = args.betas[beta_index]
            beta1 = beta
            beta2 = beta+0.05/np.log(args.n)
            _, _, nlls = approxinf_nll(train_loader, valid_loader, args, mc, beta_index, path)

            lambda_beta1 = (nlls.mean() - (nlls * np.exp(-(beta2 - beta1) * nlls)).mean() / (np.exp(-(beta2 - beta1) * nlls)).mean()) / (1 / beta1 - 1 / beta2)
            lambdas_beta1 = np.append(lambdas_beta1, lambda_beta1)
            RLCT_estimates = np.append(RLCT_estimates, lambdas_beta1.mean())


        print('MC: {} RLCT estimate: {:.2f}'.format(mc, lambdas_beta1.mean()))

    return RLCT_estimates


def varTI(args, kwargs):

    RLCT_estimates = np.empty(0)

    # set optimal parameter
    args.betas = np.array([1 / np.log(args.n)])

    # for each MC, calculate (beta)^2 Var_w^\beta nL_n(w)
    for mc in range(0, args.MCs):

        path = './{}_sanity_check/taskid{}/img/mc{}'.format(args.VItype, args.taskid, mc)
        if not os.path.exists(path):
            os.makedirs(path)

        # draw new training-testing split
        train_loader, valid_loader, test_loader, input_dim, output_dim, loss, true_RLCT = get_dataset_by_id(args, kwargs)
        _, var_nll , _ = approxinf_nll(train_loader, valid_loader, args, mc, 0, path)
        RLCT_estimates = np.append(RLCT_estimates, var_nll/(np.log(args.n)**2))
        print('MC: {} (beta)^2 Var_w^\beta nL_n(w): {:.2f}'.format(mc, var_nll/(np.log(args.n)**2)))

    return RLCT_estimates


def main():

    random.seed()

    # Training settings
    parser = argparse.ArgumentParser(description='RLCT Variational Inference')

    parser.add_argument('--taskid', type=int, default=1000,
                        help='taskid from sbatch')

    parser.add_argument('--dataset', type=str, default='logistic_synthetic',
                        help='dataset name from dataset_factory.py (default: )',
                        choices=['iris_binary', 'breastcancer_binary', 'mnist_binary', 'mnist','logistic_synthetic', 'tanh_synthetic', 'reducedrank_synthetic'])

    parser.add_argument('--syntheticsamplesize', type=int, default=100,
                        help='sample size of synthetic dataset')

    # if synthetic dataset, have to provide either w_dim or dpower
    parser.add_argument('--w_dim', type=int, default=2, help='total number of parameters in model')

    parser.add_argument('--dpower', type=float, default=None,
                        help='override w_dim and sets dimension of model to n^dpower')

    parser.add_argument('--network', type=str, default='logistic',
                        help='name of network in models.py (default: logistic)',
                        choices=['ffrelu','cnn','logistic', 'tanh', 'reducedrank'])

    parser.add_argument('--bias',action='store_true', default=False, help='turn on if model should have bias terms') #only applies to logistic right now, for purpose of testing lr_synthetic

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')

    parser.add_argument('--batchsize', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')


    # variational inference

    parser.add_argument('--VItype', type=str, default='explicit',
                        help='type of variaitonal inference',
                        choices=['explicit','implicit'])

    parser.add_argument('--prior', type=str, default='gaussian', metavar='P',
                        help='prior used on model parameters (default: gaussian)',
                        choices=['gaussian', 'mixtgauss', 'conjugate', 'conjugate_known_mean'])

    parser.add_argument('--pretrainDepochs', type=int, default=100,
                        help='number of epochs to pretrain discriminator')

    parser.add_argument('--trainDepochs', type=int, default=50,
                        help='number of epochs to train discriminator for each minibatch update of generator')

    parser.add_argument('--n_hidden_D', type=int, default=128,
                        help='number of hidden units in discriminator D')

    parser.add_argument('--num_hidden_layers_D', type=int, default=1,
                        help='number of hidden layers in discriminatror D')

    parser.add_argument('--n_hidden_G', type=int, default=128,
                        help='number of hidden units in generator G')

    parser.add_argument('--num_hidden_layers_G', type=int, default=1,
                        help='number of hidden layers in generator G')

    # optimization

    parser.add_argument('--lr_primal', type=float,  default=1e-3, metavar='LR',
                        help='primal learning rate (default: 0.01)')

    parser.add_argument('--lr_dual', type=float, default=1e-3, metavar='LR',
                        help='dual learning rate (default: 0.01)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')


    # asymptotics

    parser.add_argument('--elasticnet_alpha', type=float, default=0.5,
                        help='penalty factor for elastic net in lsfit of lambda, 0.0 for ols and 1.0 for elastic net')

    parser.add_argument('--lambda_asymptotic', type=str, default='thm4',
                        help='which asymptotic characterisation of lambda to use',
                        choices=['thm4', 'thm4_average', 'cor3','varTI'])

    parser.add_argument('--beta_auto_liberal', action="store_true", default=False,
                        help='flag to turn ON calculate optimal (liberal) range of betas based on sample size')

    parser.add_argument('--beta_auto_conservative', action="store_true", default=False,
                        help='flag to turn ON calculate optimal (conservative) range of betas based on sample size')

    parser.add_argument('--beta_auto_oracle', action="store_true", default=False,
                        help='flag to turn ON calculate optimal (oracle) range of betas based on sample size')

    parser.add_argument('--betasbegin', type=float, default=0.5,
                        help='where beta range should begin')

    parser.add_argument('--betasend', type=float, default=1.5,
                        help='where beta range should end')

    parser.add_argument('--betalogscale', action="store_true",
                        help='turn on if beta should be on 1/log n scale')

    parser.add_argument('--numbetas', type=int,  default=2,
                        help='how many betas should be swept between betasbegin and betasend')

    # as high as possible
    # parser.add_argument('--epsilon_mc', type=int, default=10,
    #                     help='number of draws for estimating E_\epsilon')

    parser.add_argument('--MCs', type=int, default=1,
                        help='number of times to split into train-test')

    parser.add_argument('--R', type=int, default=200,
                        help='number of MC draws from approximate posterior (default:200)')

    # visualization/logging

    parser.add_argument('--wandb_on', action="store_true",
                        help='use wandb to log experiment')

    parser.add_argument('--tsne_viz', action="store_true",
                        help='use tsne visualization of generator')

    parser.add_argument('--posterior_viz', action="store_true", default=True,
                        help='should only use with lr_synthetic and w_dim = 2, bias = False')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument("--mode", default='client')

    parser.add_argument("--port", default=62364)

    args = parser.parse_args()


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

    if args.dataset in ['logistic_synthetic','tanh_synthetic','reducedrank_synthetic']:
        if args.w_dim is None and args.dpower is None:
            parser.error('w_dim or dpower is necessary for synthetic data')
        if args.posterior_viz:
            if args.w_dim is None or (args.bias==True):
                parser.error('posterior visualisation only supports args.w_dim = 2 and args.bias = False')
            if args.dataset not in ['logistic_synthetic']:
                parser.error('posterior visualisation only supports lr_synthetic at the moment')

    # set true network weights for synthetic dataset
    if args.dataset == 'logistic_synthetic':

        if args.dpower is None:
            if args.bias:
                input_dim = args.w_dim
            else:
                input_dim = args.w_dim-1
        else:
            input_dim = int(np.power(args.syntheticsamplesize, args.dpower))

        args.w_0 = torch.randn(input_dim, 1)

        if args.bias:
            args.b = torch.randn(1)
        else:
            args.b = torch.tensor([0.0])

        if args.posterior_viz:
            args.w_0 = torch.Tensor([[0.5], [1]])
            args.b = torch.tensor([0.0])

    elif args.dataset == 'tanh_synthetic':

        # TODO: add option when dpower is None and w_dim is given instead
        H = int(np.power(args.syntheticsamplesize, args.dpower)*0.5) #number of hidden unit
        args.H = H
        args.a_params = torch.zeros([H, 1], dtype=torch.float32) # H * input_dim
        args.b_params = torch.zeros([1, H], dtype=torch.float32) # output_dim * H

    elif args.dataset == 'reducedrank_synthetic':

        # TODO: add option when dpower is None and w_dim is given instead
        #suppose input_dimension=output_dimension + 3, H = output_dimension, H is number of hidden nuit
        #solve the equation (input_dimension + output_dimension)*H = np.power(args.syntheticsamplesize, args.dpower) to get output_dimension, then input_dimension, and H
        output_dim = int((-3 + math.sqrt(9 + 4*2*np.power(args.syntheticsamplesize, args.dpower)))/4)
        H = output_dim
        input_dim = output_dim + 3
        args.H = H
        args.a_params = torch.cat((torch.eye(H),torch.ones([H, input_dim-H],dtype=torch.float32)), 1) # H * input_dim
        args.b_params = torch.eye(output_dim) # output_dim * H
        #in this case, the rank r for args.b_params*args.a_params is H, output_dim + H < input_dim + r is satisfied

    # draw a training-validation-testing split just to get some necessary parameters
    train_loader, valid_loader, test_loader, input_dim, output_dim, loss_criterion, trueRLCT = get_dataset_by_id(args, kwargs)
    args.n = len(train_loader.dataset)
    args.input_dim = input_dim
    args.output_dim = output_dim
    args.loss_criterion = loss_criterion
    args.trueRLCT = trueRLCT

    # retrieve model
    model, w_dim = retrieve_model(args)
    args.model = model
    args.w_dim = w_dim

    # set range of betas
    # set_betas(args)
    args.betas = np.array([1, 1])

    print(vars(args))

    if args.lambda_asymptotic == 'thm4':

        RLCT_estimates_robust, RLCT_estimates_OLS = lambda_thm4(args, kwargs)
        results = dict({
            "true_RLCT": args.trueRLCT,
            "d_on_2": args.w_dim/2,
            "RLCT estimates (robust)": RLCT_estimates_robust,
            "mean RLCT estimates (robust)": RLCT_estimates_robust.mean(),
            "std RLCT estimates (robust)": RLCT_estimates_robust.std(),
            "RLCT estimates (OLS)": RLCT_estimates_OLS,
            "mean RLCT estimates (OLS)": RLCT_estimates_OLS.mean(),
            "std RLCT estimates (OLS)": RLCT_estimates_OLS.std(),
        })

    elif args.lambda_asymptotic == 'thm4_average':

        RLCT_estimate_robust, RLCT_estimate_OLS = lambda_thm4average(args, kwargs)
        results = dict({
            "true_RLCT": [args.trueRLCT],
            "d on 2": [args.w_dim/2],
            "RLCT estimate (robust)": [RLCT_estimate_robust],
            "RLCT estimate (OLS)": [RLCT_estimate_OLS]
        }) # since all scalar, need to listify to avoid error in from_dict

    elif args.lambda_asymptotic == 'cor3':

        RLCT_estimates = lambda_cor3(args, kwargs)
        results = dict({
            "true_RLCT": args.trueRLCT,
            "d on 2": args.w_dim/2,
            "RLCT estimates": RLCT_estimates,
            "mean RLCT estimates": RLCT_estimates.mean(),
            "std RLCT estimates": RLCT_estimates.std()
        })

    elif args.lambda_asymptotic == 'varTI':

        RLCT_estimates = varTI(args, kwargs)
        results = dict({
            "true_RLCT": args.trueRLCT,
            "d on 2": args.w_dim/2,
            "RLCT estimates": RLCT_estimates,
            "mean RLCT estimates": RLCT_estimates.mean(),
            "std RLCT estimates": RLCT_estimates.std()
        })

    # just dump performance metrics to wandb, it already saves configuration
    if args.wandb_on:
        wandb.log(results)

    # save locally
    path = './{}_sanity_check/taskid{}/'.format(args.VItype, args.taskid)
    if not os.path.exists(path):
        os.makedirs(path)

    args_dict = vars(args)
    if args.dataset == 'logistic_synthetic':
        for key in ['w_0', 'b', 'loss_criterion', 'model', 'betas']:
            del args_dict[key]
    if args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
        for key in ['H', 'a_params', 'b_params', 'loss_criterion', 'model', 'betas']:
            del args_dict[key]

    results_args = pd.concat([pd.DataFrame.from_dict(results), pd.concat([pd.DataFrame.from_dict(args_dict, orient='index').transpose()]*args.MCs, ignore_index=True)], axis=1)

    results_args.to_csv('./{}_sanity_check/taskid{}/configuration_plus_results.csv'.format(args.VItype, args.taskid), index=None, header=True)


if __name__ == "__main__":
    main()


