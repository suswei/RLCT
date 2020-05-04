from __future__ import print_function

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import argparse
import random
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from random import randint
import scipy.stats as st

from dataset_factory import get_dataset_by_id
from explicit_vi import *
from implicit_vi import *
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
        fig.write_image('./{}/posteior_betaind{}.png'.format(saveimgpath, beta_index))


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
        plt.savefig('./{}/tsne_betaind{}.png'.format(saveimgpath, beta_index))
    plt.close()


# Approximate inference estimate of E_w^\beta [nL_n(w)], Var_w^\beta [nL_n(w)] based on args.R sampled w_r^*
def approxinf_nll(train_loader, valid_loader, args, mc, beta_index, saveimgpath):

    args.epsilon_dim = args.w_dim
    args.epsilon_mc = args.batchsize  # TODO: overwriting args parser input

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

        train_loader, valid_loader, test_loader = get_dataset_by_id(args, kwargs)

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
            train_loader, valid_loader, test_loader= get_dataset_by_id(args, kwargs)
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
        train_loader, valid_loader, test_loader = get_dataset_by_id(args, kwargs)

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
        train_loader, valid_loader, test_loader= get_dataset_by_id(args, kwargs)
        _, var_nll , _ = approxinf_nll(train_loader, valid_loader, args, mc, 0, path)
        RLCT_estimates = np.append(RLCT_estimates, var_nll/(np.log(args.n)**2))
        print('MC: {} (beta)^2 Var_w^\beta nL_n(w): {:.2f}'.format(mc, var_nll/(np.log(args.n)**2)))

    return RLCT_estimates


def main():

    random.seed()

    # Training settings
    parser = argparse.ArgumentParser(description='RLCT Variational Inference')

    parser.add_argument('--taskid', type=int, default=1000+randint(0, 1000),
                        help='taskid from sbatch')

    parser.add_argument('--dataset', type=str, default='logistic_synthetic',
                        help='dataset name from dataset_factory.py (default: )',
                        choices=['iris_binary', 'breastcancer_binary', 'mnist_binary', 'mnist', 'logistic_synthetic', 'tanh_synthetic', 'reducedrank_synthetic'])

    parser.add_argument('--syntheticsamplesize', type=int, default=100,
                        help='sample size of synthetic dataset')

    # if synthetic dataset, have to provide either w_dim or dpower
    parser.add_argument('--w_dim', type=int, help='total number of parameters in model')

    parser.add_argument('--dpower', type=float,
                        help='would set total number of model parameters to n^dpower')

    parser.add_argument('--network', type=str, default='logistic',
                        help='name of network in models.py (default: logistic)',
                        choices=['ffrelu','cnn','logistic', 'tanh', 'reducedrank'])

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

    parser.add_argument('--betalogscale', action="store_true", default=False,
                        help='turn on if beta should be on 1/log n scale')

    parser.add_argument('--numbetas', type=int,  default=20,
                        help='how many betas should be swept between betasbegin and betasend')

    # as high as possible
    # parser.add_argument('--epsilon_mc', type=int, default=10,
    #                     help='number of draws for estimating E_\epsilon')

    parser.add_argument('--MCs', type=int, default=1,
                        help='number of times to split into train-test')

    parser.add_argument('--R', type=int, default=200,
                        help='number of MC draws from approximate posterior (default:200)')

    # visualization/logging
    parser.add_argument('--notebook', action="store_true", default=False,
                        help='turn on for plotly notebook render')

    parser.add_argument('--tsne_viz', action="store_true", default=False,
                        help='use tsne visualization of generator')

    parser.add_argument('--posterior_viz', action="store_true",
                        help='should only use with lr_synthetic, w_dim = 2, bias = False')

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
            if (args.w_dim != 2) or (args.bias==True):
                parser.error('posterior visualisation only supports args.w_dim = 2 and args.bias = False')


    # set necessary parameters related to dataset in args
    get_dataset_by_id(args, kwargs)

    # retrieve model
    args.model, args.w_dim = retrieve_model(args)

    # get grid of betas for RLCT asymptotics
    set_betas(args)

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


