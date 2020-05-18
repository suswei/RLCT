# The setup here is similar to Table 8.1 in Watanabe textbook. I don't use his prior for A and B however. He also never specifies how he chose A_0 and B_0

from __future__ import print_function

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

import sys
import numpy as np

sys.path.append('../')
from main import *


class Args:

    dataset = 'reducedrank_synthetic'
    sanity_check = True
    syntheticsamplesize = 1000

    network = 'reducedrank'
    VItype = 'implicit'
    batchsize = 100
    epochs = 200
    epsilon_mc = 100
    pretrainDepochs = 100
    trainDepochs = 50
    n_hidden_D = 128
    num_hidden_layers_D = 2
    n_hidden_G = 256
    num_hidden_layers_G = 1

    lr_primal = 1e-3
    lr_dual = 5e-2

    beta_auto_liberal = False
    beta_auto_conservative = False
    beta_auto_oracle = False
    betasbegin = 0.1
    betasend = 1.5
    betalogscale = True
    numbetas = 10

    elasticnet_alpha = 1.0
    R = 200

    MCs = 10

    log_interval = 50

    cuda = False


args=Args()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def train(args, train_loader, valid_loader):

    # get a grid of inverse temperatures [beta_1/log n, \ldots, beta_k/log n]
    set_betas(args)

    mc = 1
    saveimgpath = None
    nll_betas_implicit = np.empty(0)

    for beta_index in range(args.betas.shape[0]):

        # train implicit variational inference
        print('Begin training IVI')
        G = train_implicitVI(train_loader, valid_loader, args, mc, beta_index, saveimgpath)
        print('Finished training IVI')
        nllw_array_implicit = approxinf_nll_implicit(train_loader, G, args)
        nllw_mean_implicit = sum(nllw_array_implicit)/len(nllw_array_implicit)
        nll_betas_implicit = np.append(nll_betas_implicit, nllw_mean_implicit)

    ivi_robust, ivi_ols = lsfit_lambda(nll_betas_implicit, args, saveimgname=None)

    return ivi_robust


def compute_predictive_dist(args, G, loader):

    R = 1000
    eps = torch.randn(R, args.epsilon_dim)
    sampled_weights = G(eps)
    wholex = loader.dataset[:][0]
    wholey = loader.dataset[:][1]
    list_of_param_dicts = weights_to_dict(args, sampled_weights)

    pred_logprob = 0
    for param_dict in list_of_param_dicts:

        mean = torch.matmul(torch.matmul(wholex, param_dict['a']), param_dict['b'])
        pred_logprob += -(args.output_dim*np.log(2 * np.pi) + torch.norm(wholey-mean,dim=1)**2) / 2

    return pred_logprob/R


def compute_Bg(pred_logprob,loader,args):

    wholex = loader.dataset[:][0]
    wholey = loader.dataset[:][1]
    mean = torch.matmul(torch.matmul(wholex, args.a_params), args.b_params)
    mean_dev = torch.norm(wholey-mean,dim=1)**2
    true_logprob = -(args.output_dim * np.log(2 * np.pi) + mean_dev) / 2

    return (true_logprob-pred_logprob).mean()


args.input_dim = 6
args.output_dim = 6

args.a_params = torch.randn(args.input_dim,3)
args.b_params = torch.randn(3,args.output_dim)
H0 = torch.matrix_rank(torch.matmul(args.a_params,args.b_params))

Hrange = range(3, 6)
results = []
for H in Hrange:

        args.H = H
        args.model, args.w_dim = retrieve_model(args)
        args.epsilon_dim = args.w_dim

        Bg = np.empty(args.MCs)
        rlct = np.empty(args.MCs)

        for mc in range(0, args.MCs):

            train_loader, valid_loader, test_loader = get_dataset_by_id(args, kwargs)

            args.betas = [1.0]
            beta_index = 0
            G = train_implicitVI(train_loader, valid_loader, args, mc, beta_index, saveimgpath=None)
            with torch.no_grad():
                pred = compute_predictive_dist(args, G, test_loader)
                Bg[mc] = compute_Bg(pred, test_loader, args)

            rlct[mc] = train(args, train_loader, valid_loader)
            print('reduced rank regression model H {}: mc {}: Bg {} rlct {}'.format(H,mc, Bg[mc], rlct[mc]))

        print('H: {}'.format(H))
        print('E_n Bg(n): {}'.format(Bg.mean()))
        print('hat RLCT/n: {}'.format(rlct.mean() / args.syntheticsamplesize))
        results.append({'H':H,'E_n Bg(n)': Bg.mean(), 'hat RLCT/n': rlct.mean()/ args.syntheticsamplesize})

with open('generalization_rr.pkl', 'wb') as f:
    pickle.dump(results, f)