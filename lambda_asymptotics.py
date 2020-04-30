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
from RLCT_helper import retrieve_model, load_minibatch, randn, lsfit_lambda, EarlyStopping
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from random import randint
from main import approxinf_nll

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