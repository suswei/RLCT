from __future__ import print_function

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import argparse
import random
# from sklearn.manifold import TSNE
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

# TODO: Add 3D support
def posterior_viz(train_loader, sampled_weights, args, beta_index, saveimgpath):

    wmin = -3
    wmax = 3
    wspace = 50
    wrange = np.linspace(wmin, wmax, wspace)
    w = np.repeat(wrange[:, None], wspace, axis=1)
    w = np.concatenate([[w.flatten()], [w.T.flatten()]])

    # prior
    logprior = -(w ** 2).sum(axis=0) / 2

    # true unnormalised posterior at inverse temp
    logpost = torch.zeros(wspace * wspace)
    for i in range(wspace * wspace):
        current_w = torch.from_numpy(w[:,i]).float().unsqueeze(dim=0) # convert to torch tensor of shape [1,w_dim]
        param_dict = weights_to_dict(args, current_w)[0]
        logpost[i] = -args.betas[beta_index] * calculate_nllsum_paramdict(args, train_loader.dataset[:][1], train_loader.dataset[:][0], param_dict)
    logpost = logpost.detach().numpy() + logprior

    # kde sampled_weights
    kernel = st.gaussian_kde([sampled_weights[:,0].detach().numpy(), sampled_weights[:,1].detach().numpy()])
    f = np.reshape(kernel(w).T, [wspace,wspace])

    fig = make_subplots(rows=1, cols=3, subplot_titles=('(unnormalised) prior',
                                                        '(unnormalised) posterior',
                                                        'kde of sampled weights'))
    fig.add_trace(
        go.Contour(
            z=np.exp(logprior.reshape(wspace, wspace)),
            x=wrange,  # horizontal axis
            y=wrange
        ),  # vertical axis
        row=1, col=1
    )
    fig.add_trace(
        go.Contour(
            z=np.exp(logpost.reshape(wspace, wspace).T),
            x=wrange,  # horizontal axis
            y=wrange
        ),  # vertical axis
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=sampled_weights[:, 0].detach().numpy(),
            y=sampled_weights[:, 1].detach().numpy(),
            mode='markers',
            name='Sampled Data',
            marker=dict(size=3,opacity=0.6)
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Contour(
            z=f.T,
            x=wrange,  # horizontal axis
            y=wrange),  # vertical axis
            row=1, col=3
    )
    fig.update_layout(title_text='{}, {}, beta {}'.format(args.dataset, args.VItype, args.betas[beta_index]))
    if args.notebook:
        fig.show(renderer='notebook')
    else:
        fig.show()
    if saveimgpath is not None:
        fig.write_image('./{}/posteior_betaind{}'.format(saveimgpath, beta_index))


def tsne_viz(sampled_weights,args,beta_index,saveimgpath):

    if args.dataset == 'logistic_synthetic':
        true_weight = torch.cat((args.w_0.reshape(1, args.input_dim), args.b.reshape(1, 1)), 1)
    elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
        true_weight = torch.cat((args.a_params.reshape(1, (args.a_params.shape[0] * args.a_params.shape[1])), args.b_params.reshape(1, (args.b_params.shape[0] * args.b_params.shape[1]))), 1)

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
    if saveimgpath is not None:
        plt.savefig('./{}/tsne_betaind{}'.format(saveimgpath, beta_index))
    plt.close()


# Approximate inference estimate of E_w^\beta [nL_n(w)], Var_w^\beta [nL_n(w)] based on args.R sampled w_r^*
def approxinf_nll(train_loader, valid_loader, args, mc, beta_index, saveimgpath):

    args.epsilon_dim = args.w_dim
    args.epsilon_mc = args.batchsize  # TODO: is epsilon_mc sensitive?

    if args.VItype == 'implicit':

        G = train_implicitVI(train_loader, valid_loader, args, mc, beta_index, saveimgpath)
        nllw_array = approxinf_nll_implicit(train_loader, G, args)

        # visualize generator G
        with torch.no_grad():

            if args.tsne_viz or args.posterior_viz:
                eps = torch.randn(100, args.epsilon_dim)
                sampled_weights = G(eps)

            # if args.tsne_viz:
            #     tsne_viz(w_sampled_from_G, args, beta_index, saveimgpath)

            if args.posterior_viz:
                posterior_viz(train_loader, sampled_weights, args, beta_index, saveimgpath)


    elif args.VItype == 'explicit':

        var_model = train_explicitVI(train_loader, valid_loader, args, mc, beta_index, True, saveimgpath)
        nllw_array = approxinf_nll_explicit(train_loader, var_model, args)

        with torch.no_grad():

            if args.tsne_viz or args.posterior_viz:
                sampled_weights = sample_EVI(var_model, args)

            # if args.tsne_viz:
            #     tsne_viz(sampled_weights, args, beta_index, saveimgpath)

            if args.posterior_viz:
                posterior_viz(train_loader, sampled_weights, args, beta_index, saveimgpath)

    return nllw_array.mean(), nllw_array.var(), nllw_array


def lambda_asymptotics(args, kwargs):

    nlls_mean = np.empty((args.MCs, args.numbetas))
    # theorem 4
    RLCT_estimates_ols = np.empty(0)
    RLCT_estimates_robust = np.empty(0)

    for mc in range(0, args.MCs):
        # draw new training-testing split
        train_loader, valid_loader, test_loader = get_dataset_by_id(args, kwargs)
        for beta_index in range(args.numbetas):
            print('Starting mc {}/{}, beta {}/{}'.format(mc, args.MCs, beta_index, args.numbetas))
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

        with open('{}/results.pkl'.format(args.path), 'wb') as f:
            pickle.dump(results_dict, f)

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

    elif args.dataset == 'tanh_synthetic':  # "Resolution of Singularities ... for Layered Neural Network" Aoyagi and Watanabe

        if args.dpower is None:
            args.H = int(args.w_dim/2)
        else:
            args.H = int(np.power(args.syntheticsamplesize, args.dpower)*0.5) #number of hidden unit
        args.a_params = torch.zeros([1, args.H], dtype=torch.float32)
        args.b_params = torch.zeros([args.H, 1], dtype=torch.float32)

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

    elif args.dataset == 'ffrelu_synthetic':

        args.input_dim = 1
        args.output_dim = 1
        args.dataset = 'ffrelu_synthetic'
        args.network = 'ffrelu'

        # set state dictionary of ffrelu

        ffrelu_true = models.ffrelu(args.input_dim, args.output_dim, 2, 4)

        args.true_mean = ffrelu_true


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
                        choices=['ffrelu','cnn','logistic', 'tanh', 'reducedrank'])

    parser.add_argument('--H1',type=int, help = 'number of hidden units in layer 1 of ffrelu')
    parser.add_argument('--H2',type=int, help = 'number of hidden units in layer 2 of ffrelu')

    parser.add_argument('--bias',action='store_true', default=False, help='turn on if model should have bias terms') #only applies to logistic right now, for purpose of testing lr_synthetic

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')

    parser.add_argument('--batchsize', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')


    # variational inference

    parser.add_argument('--VItype', type=str, default='implicit',
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

    parser.add_argument('--betasbegin', type=float, default=0.5,
                        help='where beta range should begin')

    parser.add_argument('--betasend', type=float, default=1.5,
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
    path = './{}_sanity_check/taskid{}'.format(args.VItype, args.taskid)
    if not os.path.exists(path):
        os.makedirs(path)

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
    with open('{}/config.pkl'.format(path), 'wb') as f:
        pickle.dump(args_dict, f)

    print('Starting taskid {}'.format(args.taskid))
    results_dict = lambda_asymptotics(args, kwargs)
    print(results_dict)
    print('Finished taskid {}'.format(args.taskid))

    with open('{}/results.pkl'.format(path), 'wb') as f:
        pickle.dump(results_dict, f)


    # pickle.load(open('{}/config.pkl'.format(path),"rb"))
    # pickle.load(open('{}/results.pkl'.format(path),"rb"))


if __name__ == "__main__":
    main()


