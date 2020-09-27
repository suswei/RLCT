from __future__ import print_function

# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
import os
import argparse
import random
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from random import randint
import scipy.stats as st
import pickle
import math
import logging
import sys

from dataset_factory import get_dataset_by_id
from implicit_vi import *
from explicit_vi import *
from RLCT_helper import *
from mcmc_helper import *
from models import *
from visualization import *


# Approximate inference estimate of E_w^\beta [nL_n(w)], Var_w^\beta [nL_n(w)] based on args.R sampled w_r^*
def approxinf_nll(train_loader, valid_loader, args, mc, beta_index, saveimgpath):

    args.epsilon_dim = args.w_dim
    # args.epsilon_mc = args.batchsize  # TODO: is epsilon_mc sensitive?

    if args.posterior_method == 'mcmc':

        wholex = train_loader.dataset[:][0]
        wholey = train_loader.dataset[:][1]

        beta = args.betas[beta_index]

        # TODO: this should be flexible enough for pyro_blah
        start = time.time()
        kernel = NUTS(pyro_ffrelu, adapt_step_size=True)
        mcmc = MCMC(kernel, num_samples=args.num_samples, warmup_steps=args.num_warmup)
        mcmc.run(wholex, wholey, args.H, beta)
        print('\nMCMC elapsed time:', time.time() - start)

        nll_mean, nll_var, nll_array = expected_nll_posterior_relu(mcmc.get_samples(), args, wholex, wholey)

    elif args.posterior_method == 'implicit':

        G = train_implicitVI(train_loader, valid_loader, args, mc, beta_index, saveimgpath)
        nll_array = approxinf_nll_implicit(train_loader, G, args)

        # visualize generator G
        with torch.no_grad():

            if args.tsne_viz or args.posterior_viz:
                eps = torch.randn(100, args.epsilon_dim)
                sampled_weights = G(eps)

            # if args.tsne_viz:
            #     tsne_viz(w_sampled_from_G, args, beta_index, saveimgpath)

            if args.posterior_viz:
                posterior_viz(train_loader, sampled_weights, args, beta_index, saveimgpath)

        nll_mean = sum(nll_array) / len(nll_array)
        nll_var = sum((x - nll_mean) ** 2 for x in nll_array) / len(nll_array)

    elif args.posterior_method == 'explicit':

        var_model = train_explicitVI(train_loader, valid_loader, args, mc, beta_index, True, saveimgpath)
        nll_array = approxinf_nll_explicit(train_loader, var_model, args)

        with torch.no_grad():

            if args.tsne_viz or args.posterior_viz:
                sampled_weights = sample_EVI(var_model, args)

            # if args.tsne_viz:
            #     tsne_viz(sampled_weights, args, beta_index, saveimgpath)

            if args.posterior_viz:
                posterior_viz(train_loader, sampled_weights, args, beta_index, saveimgpath)

        nll_mean = sum(nll_array) / len(nll_array)
        nll_var = sum((x - nll_mean) ** 2 for x in nll_array) / len(nll_array)

    return nll_mean, nll_var, nll_array


def lambda_asymptotics(args, kwargs):

    nlls_mean = np.empty((args.MCs, args.numbetas))
    # theorem 4
    RLCT_estimates_ols = np.empty(0)
    RLCT_estimates_robust = np.empty(0)

    for mc in range(0, args.MCs):
        # draw new training-testing split
        train_loader, valid_loader, test_loader = get_dataset_by_id(args, kwargs)
        for beta_index in range(args.numbetas):
            print('Starting mc {}/{}, beta {}/{}'.format(mc+1, args.MCs, beta_index+1, args.numbetas))
            nll_mean, _, _ = approxinf_nll(train_loader, valid_loader, args, mc, beta_index, None)
            nlls_mean[mc, beta_index] = nll_mean

        saveimgname = '{}/thm4_lsfit_mc{}'.format(args.path,mc)
        robust, ols = lsfit_lambda(nlls_mean[mc, :], args, saveimgname)
        RLCT_estimates_robust = np.append(RLCT_estimates_robust, robust)
        RLCT_estimates_ols = np.append(RLCT_estimates_ols, ols)

        results_dict = {'rlct robust thm4 array': RLCT_estimates_robust,
                        'rlct robust thm4 mean': RLCT_estimates_robust.mean(),
                        'rlct robust thm4 std': RLCT_estimates_robust.std(),
                        'rlct ols thm4 array': RLCT_estimates_ols,
                        'rlct ols thm4 mean': RLCT_estimates_ols.mean(),
                        'rlct ols thm4 std': RLCT_estimates_ols.std()}

        torch.save(results_dict, '{}/results.pt'.format(args.path))


    # theorem 4 average
    # nlls_mean.mean(axis=0) shape should be 1, numbetas
    if args.MCs > 1:
        saveimgname = '{}/thm4_average_lsfit'.format(args.path)
        robust, ols = lsfit_lambda(nlls_mean.mean(axis=0), args, saveimgname)
        results_dict.update({'rlct robust thm4 average': robust, 'rlct ols thm4 average': ols})

    # variance thermodynamic integration Imai
    # RLCT_estimates = np.empty(0)
    # args.betas = np.array([1 / np.log(args.n)])
    # for mc in range(0, args.MCs):
    #     print('Starting mc {}/{}: var TI'.format(mc, args.MCs, beta_index, args.numbetas))
    #     # draw new training-testing split
    #     train_loader, valid_loader, test_loader = get_dataset_by_id(args, kwargs)
    #     _, var_nll, _ = approxinf_nll(train_loader, valid_loader, args, mc, 0, None)
    #     RLCT_estimates = np.append(RLCT_estimates, var_nll/(np.log(args.n)**2))
    #
    # results_dict.update({'rlct var TI array': RLCT_estimates,
    #                      'rlct var TI mean': RLCT_estimates.mean(),
    #                      'rlct var TI std': RLCT_estimates.std()})

    return results_dict


# set up true parameters for synthetic datasets
def setup_w0(args):

    if args.dataset == 'logistic_synthetic':

        if args.dpower is None:
            if args.bias:
                args.input_dim = args.w_dim - 1
            else:
                args.input_dim = args.w_dim
        else:
            args.input_dim = int(np.power(args.syntheticsamplesize, args.dpower))

        args.w_0 = torch.randn(args.input_dim, 1)

        if args.bias:
            args.b = torch.randn(1)
        else:
            args.b = torch.tensor([0.0])

        if args.posterior_viz:
            args.w_0 = torch.Tensor([[0.5], [1]])
            args.b = torch.tensor([0.0])

        args.output_dim = 1

        if args.sanity_check:
            args.network = 'logistic'

    elif args.dataset == 'tanh_synthetic':  # "Resolution of Singularities ... for Layered Neural Network" Aoyagi and Watanabe

    # univariate input and univariate output
    # mean function h(x,w) = \sum_{k=1}^H a_k tanh(b_k x + c_k)
    # Suppose the true distribution q(y|x) has mean 0, i.e. H=0

        if args.dpower is None:
            args.H = int(args.w_dim/3)
        else:
            args.H = int(np.power(args.syntheticsamplesize, args.dpower)*0.5) #number of hidden unit

        if args.sanity_check:
            args.network = 'tanh'
            if args.posterior_method == 'mcmc':
                args.network = 'pyro_tanh'

    elif args.dataset == 'reducedrank_synthetic':

        # TODO: design A_0, B_0 so the loci are equivalent, was suggested to make B_0A_0 surjective
        # suppose input_dimension=output_dimension + 3, H = output_dimension, H is number of hidden nuit
        # solve the equation (input_dimension + output_dimension)*H = np.power(args.syntheticsamplesize, args.dpower) to get output_dimension, then input_dimension, and H
        if args.dpower is None:
            args.output_dim = int((-3 + math.sqrt(9 + 4 * 2 * args.w_dim)) / 4)
        else:
            args.output_dim = int((-3 + math.sqrt(
                9 + 4 * 2 * np.power(args.syntheticsamplesize, args.dpower))) / 4)  # TODO: can easily be zero

        args.H = args.output_dim
        args.input_dim = args.output_dim + 3
        args.a_params = torch.transpose(
            torch.cat((torch.eye(args.H), torch.ones([args.H, args.input_dim - args.H], dtype=torch.float32)), 1), 0,
            1)  # input_dim * H
        args.b_params = torch.eye(args.output_dim)

        if args.w_dim == 2:
            args.a_params = torch.Tensor([1.0]).reshape(1, 1)
            args.b_params = torch.Tensor([1.0]).reshape(1, 1)
            args.input_dim = 1
            args.output_dim = 1
            args.H = 1
        # in this case, the rank r for args.b_params*args.a_params is H, output_dim + H < input_dim + r is satisfied

        if args.sanity_check:
            args.network = 'reducedrank'

    elif args.dataset == 'ffrelu_synthetic':

        args.input_dim = 10
        args.output_dim = 10
        args.H = 10
        # Currently hardcoded true hidden unit numbers
        args.true_mean = models.ffrelu(args.input_dim, args.output_dim, 4, 2)

        if args.sanity_check:
            args.network = 'ffrelu'
            if args.posterior_method == 'mcmc':
                args.network = 'pyro_ffrelu'


def main():

    # random.seed()

    # Training settings
    parser = argparse.ArgumentParser(description='RLCT Variational Inference')

    parser.add_argument('--taskid', type=int, default=1000+randint(0, 1000),
                        help='taskid from sbatch')

    parser.add_argument('--dataset', type=str, default='logistic_synthetic',
                        help='dataset name from dataset_factory.py (default: )',
                        choices=['iris_binary', 'breastcancer_binary', 'mnist_binary', 'mnist',
                                 'logistic_synthetic',
                                 'tanh_synthetic',
                                 'reducedrank_synthetic',
                                 'ffrelu_synthetic'])

    parser.add_argument('--syntheticsamplesize', type=int, default=100,
                        help='sample size of synthetic dataset')

    # if synthetic dataset, have to provide either w_dim or dpower
    parser.add_argument('--w_dim', type=int, help='total number of parameters in model')

    parser.add_argument('--dpower', type=float,
                        help='would set total number of model parameters to n^dpower')

    parser.add_argument('--sanity_check', action='store_true', default=False,
                    help='turn on if network should match synthetic generation')  # only applies to logistic right now, for purpose of testing lr_synthetic

    parser.add_argument('--network', type=str, default='logistic',
                        help='name of network in models.py (default: logistic)',
                        choices=['ffrelu','cnn','logistic', 'tanh', 'reducedrank','pyro_tanh','pyro_ffrelu'])

    parser.add_argument('--H1',type=int, help = 'number of hidden units in layer 1 of ffrelu')
    parser.add_argument('--H2',type=int, help = 'number of hidden units in layer 2 of ffrelu')

    parser.add_argument('--bias',action='store_true', default=False, help='turn on if model should have bias terms') #only applies to logistic right now, for purpose of testing lr_synthetic

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')

    parser.add_argument('--batchsize', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')

    # mcmc
    parser.add_argument("--num-samples", nargs="?", default=20000, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=10000, type=int)

    # variational inference

    parser.add_argument('--posterior_method', type=str, default='mcmc',
                        help='method for posterior estimation',
                        choices=['mcmc','explicit','implicit'])

    parser.add_argument('--prior', type=str, default='gaussian', metavar='P',
                        help='prior used on model parameters (default: gaussian)',
                        choices=['gaussian', 'mixtgauss', 'conjugate', 'conjugate_known_mean'])

    parser.add_argument('--epsilon_mc', type=int, default=100, help='used in IVI')

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

    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                          help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')

    # asymptotics

    parser.add_argument('--elasticnet_alpha', type=float, default=0.5,
                        help='penalty factor for elastic net in lsfit of lambda, 0.0 for ols and 1.0 for elastic net')

    parser.add_argument('--beta_auto_liberal', action="store_true", default=False,
                        help='flag to turn ON calculate optimal (liberal) range of betas based on sample size')

    parser.add_argument('--beta_auto_conservative', action="store_true", default=False,
                        help='flag to turn ON calculate optimal (conservative) range of betas based on sample size')

    parser.add_argument('--beta_auto_oracle', action="store_true", default=False,
                        help='flag to turn ON calculate optimal (oracle) range of betas based on sample size')

    parser.add_argument('--betasbegin', type=float, default=0.01,
                        help='where beta range should begin')

    parser.add_argument('--betasend', type=float, default=2.0,
                        help='where beta range should end')

    parser.add_argument('--betalogscale', action="store_true", default=False,
                        help='turn on if beta should be on 1/log n scale')

    parser.add_argument('--betanscale', action="store_true", default=False,
                        help='turn on if beta should be on 1/ n scale')

    parser.add_argument('--numbetas', type=int,  default=20,
                        help='how many betas should be swept between betasbegin and betasend')


    parser.add_argument('--MCs', type=int, default=1,
                        help='number of times to split into train-test')

    parser.add_argument('--R', type=int, default=200,
                        help='number of MC draws from approximate posterior (default:200)')


    # visualization/logging
    parser.add_argument('--notebook', action="store_true", default=False,
                        help='turn on for plotly notebook render')

    parser.add_argument('--tsne_viz', action="store_true", default=False,
                        help='use tsne visualization of generator')

    parser.add_argument('--posterior_viz', action="store_true", default=False,
                        help='should only use with lr_synthetic, w_dim = 2, bias = False')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')

    parser.add_argument("--mode", default='client')

    parser.add_argument("--port", default=62364)

    args = parser.parse_args()

    # log results to directory
    path = './{}_sanity_check/taskid{}'.format(args.posterior_method, args.taskid)
    if not os.path.exists(path):
        os.makedirs(path)

    # cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("args.cuda is " + str(args.cuda))
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Daniel
    # torch.manual_seed(args.seed)
    # if args.cuda:
    #    torch.cuda.manual_seed(args.seed)

    if args.posterior_method == 'mcmc':
        print('Currently only supports tanh and ffrelu dataset')

    if args.dataset in ['logistic_synthetic','tanh_synthetic','reducedrank_synthetic']:
        if args.w_dim is None and args.dpower is None:
            parser.error('w_dim or dpower is necessary for synthetic data')
        if args.posterior_viz:
            if (args.w_dim != 2) or (args.bias == True):
                parser.error('posterior visualisation only supports args.w_dim = 2 and args.bias = False')

    setup_w0(args)

    # set necessary parameters related to dataset in args
    get_dataset_by_id(args, kwargs)

    # retrieve model
    args.model, args.w_dim = retrieve_model(args)

    # get grid of betas for RLCT asymptotics
    set_betas(args)

    # record configuration for saving
    args.path = path
    args_dict = vars(args)
    print(args_dict)
    torch.save(args_dict, '{}/config.pt'.format(path))

    print('Starting taskid {}'.format(args.taskid))
    results_dict = lambda_asymptotics(args, kwargs)
    print(results_dict)
    print('Finished taskid {}'.format(args.taskid))

    torch.save(results_dict, '{}/results.pt'.format(path))


if __name__ == "__main__":
    main()


