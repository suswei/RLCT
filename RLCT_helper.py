from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.linear_model import ElasticNet
from matplotlib import pyplot as plt
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.tools import add_constant

import models


def lsfit_lambda(temperedNLL_perMC_perBeta, args, saveimgname):

    # robust ls fit
    regr = ElasticNet(random_state=0, fit_intercept=True, alpha=args.elasticnet_alpha)
    regr.fit((1 / args.betas).reshape(args.numbetas, 1), temperedNLL_perMC_perBeta)
    robust_intercept_estimate = regr.intercept_
    # slope_estimate = min(regr.coef_[0],args.w_dim/2)
    robust_slope_estimate = regr.coef_[0]

    # vanilla ols fit
    ols_model = OLS(temperedNLL_perMC_perBeta, add_constant(1 / args.betas)).fit()
    ols_intercept_estimate = ols_model.params[0]
    # slope_estimate = min(ols_model.params[1],args.w_dim/2)
    ols_slope_estimate = ols_model.params[1]

    plt.scatter(1 / args.betas, temperedNLL_perMC_perBeta, label='nll beta')
    plt.plot(1 / args.betas, robust_intercept_estimate + robust_slope_estimate * 1 / args.betas, 'g-',
             label='robust ols')
    plt.plot(1 / args.betas, ols_intercept_estimate + ols_slope_estimate * 1 / args.betas, 'b-', label='ols')

    if args.trueRLCT is None:
        plt.title("d_on_2 = {}, true lambda = {} "
                  "\n hat lambda robust = {:.1f}, hat lambda ols = {:.1f}"
                  .format(args.w_dim / 2, 'unknown', robust_slope_estimate, ols_slope_estimate), fontsize=8)
    else:
        plt.title("d_on_2 = {}, true lambda = {:.1f} "
                  "\n hat lambda robust = {:.1f}, hat lambda ols = {:.1f}"
                  .format(args.w_dim / 2, args.trueRLCT, robust_slope_estimate, ols_slope_estimate), fontsize=8)
    plt.xlabel("1/beta", fontsize=8)
    plt.ylabel("{} VI estimate of (E_data) E^beta_w [nL_n(w)]".format(args.posterior_method), fontsize=8)
    plt.legend()
    if saveimgname is not None:
        plt.savefig('{}.png'.format(saveimgname))
    plt.show()

    return robust_slope_estimate, ols_slope_estimate


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def retrieve_model(args):
    # M is the input dimension, H is hidden unit number, N is the output dimension for 'tanh_synthetic' and 'reducedrank_synthetic'
    # retrieve model
    if args.network == 'cnn':
        model = models.cnn(output_dim=args.output_dim)
        print('Error: implicit VI currently only supports logistic regression')
    if args.network == 'logistic':
        model = models.logistic(input_dim=args.input_dim, bias=args.bias)
    if args.network == 'ffrelu':
        model = models.ffrelu(input_dim=args.input_dim, output_dim=args.output_dim, H1=args.H1, H2=args.H2)
    if args.network == 'tanh':
        model = models.tanh(input_dim=args.input_dim, output_dim=args.output_dim, H=args.H)
    if args.network == 'reducedrank':
        model = models.reducedrank(input_dim=args.input_dim, output_dim=args.output_dim, H=args.H)
    if args.network == 'pyro_tanh':
        model = []
    if args.network == 'pyro_ffrelu':
        model = []

    # TODO: count parameters automatically
    if args.network == 'logistic':
        if args.dataset in ('mnist_binary', 'iris_binary', 'breastcancer_binary', 'logistic_synthetic'):
            w_dim = (args.input_dim + 1 * args.bias)
        elif args.dataset == 'mnist':
            w_dim = (args.input_dim + 1 * args.bias) * 9 / 2
    elif args.network in ['tanh', 'pyro_tanh','reducedrank']:
        w_dim = (args.input_dim + args.output_dim) * args.H
    elif args.network == 'pyro_ffrelu':
        w_dim = 5*(args.H * args.H + args.H)
    else:
        w_dim = count_parameters(model)

    return model, w_dim


def set_betas(args):
    if args.beta_auto_conservative:
        # optimal beta is given by 1/log(n)[1+U_n/\sqrt(2\lambda \log n) + o_p(1/\sqrt(2\lambda \log n) ], according to Corollary 2 of WBIC
        # since U_n is N(0,1) under certain conditions,
        # let's consider beta range [1/log(n)(1 - 1/\sqrt(2\log n)), 1/log(n)(1 + 1/\sqrt(2\log n)) ], taking the worst case for the std for U_n
        args.betas = np.linspace(1 / np.log(args.n) * (1 - 1 / np.sqrt(2 * np.log(args.n))),
                                 1 / np.log(args.n) * (1 + 1 / np.sqrt(2 * np.log(args.n))), args.numbetas)


    elif args.beta_auto_liberal:
        # optimal beta is given by 1/log(n)[1+U_n/\sqrt(2\lambda \log n) + o_p(1/\sqrt(2\lambda \log n) ], according to Corollary 2 of WBIC
        # since U_n is N(0,1) under certain conditions, for the "liberal" setting,
        # let's consider beta range [1/log(n)(1 - 1/\sqrt(2*d/2*\log n)), 1/log(n)(1 + 1/\sqrt(2*d/2*\log n)) ], taking the worst case for the std for U_n
        args.betas = np.linspace(1 / np.log(args.n) * (1 - 1 / np.sqrt(args.w_dim * np.log(args.n))),
                                 1 / np.log(args.n) * (1 + 1 / np.sqrt(args.w_dim * np.log(args.n))),
                                 args.numbetas)

    elif args.beta_auto_oracle:
        # optimal beta is given by 1/log(n)[1+U_n/\sqrt(2\lambda \log n) + o_p(1/\sqrt(2\lambda \log n) ], according to Corollary 2 of WBIC
        # since U_n is N(0,1) under certain conditions, for the "liberal" setting,
        # let's consider beta range [1/log(n)(1 - 1/\sqrt(2*d/2*\log n)), 1/log(n)(1 + 1/\sqrt(2*d/2*\log n)) ], taking the worst case for the std for U_n
        args.betas = np.linspace(1 / np.log(args.n) * (1 - 1 / np.sqrt(2 * args.trueRLCT * np.log(args.n))),
                                 1 / np.log(args.n) * (1 + 1 / np.sqrt(2 * args.trueRLCT * np.log(args.n))),
                                 args.numbetas)

    else:
        args.betas = 1 / np.linspace(1 / args.betasbegin, 1 / args.betasend, args.numbetas)
        if args.betalogscale:
            args.betas = 1 / np.linspace(np.log(args.n) / args.betasbegin, np.log(args.n) / args.betasend,
                                         args.numbetas)
        elif args.betanscale:
            args.betas = 1 / np.linspace(args.n / args.betasbegin, args.n / args.betasend,
                                         args.numbetas)


# perform some data massaging to get right dimensions
def load_minibatch(args, data, target):
    if args.dataset == 'mnist_binary':
        for ind, y_val in enumerate(target):
            target[ind] = 0 if y_val < 5 else 1

    if args.cuda:
        data, target = data.cuda(), target.cuda()

    if args.dataset in ('mnist', 'mnist_binary'):
        if args.network == 'cnn':
            data, target = Variable(data), Variable(target)
        else:
            data, target = Variable(data.view(-1, 28 * 28)), Variable(target)
    else:
        data, target = Variable(data), Variable(target)

    return data, target


def randn(shape, device):
    if device == False:
        # return torch.randn(*shape).to(device)
        return torch.randn(*shape)
    else:
        return torch.cuda.FloatTensor(*shape).normal_()


def weights_to_dict(args, sampled_weights):
    '''

    :param args:
    :param sampled_weights: [# of sampled weights, w_dim]
    :return: list of parameter dictionaries
    '''

    list_of_param_dicts = []

    for i in range(0, sampled_weights.shape[0]):

        if args.dataset == 'logistic_synthetic':

            if args.bias:
                w = sampled_weights[i, 0:(args.w_dim - 1)].reshape(args.w_dim - 1, 1)
                b = sampled_weights[i, args.w_dim - 1]
            else:
                w = sampled_weights[i, 0:(args.w_dim)].reshape(args.w_dim, 1)
                b = 0.0

            list_of_param_dicts.append({'w': w, 'b': b}.copy())

        elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:

            a_params = sampled_weights[i, 0:(args.input_dim * args.H)].reshape(args.input_dim, args.H)
            b_params = sampled_weights[i, (args.input_dim * args.H):].reshape(args.H, args.output_dim)

            list_of_param_dicts.append({'a': a_params, 'b': b_params}.copy())

        elif args.dataset == 'ffrelu_synthetic':

            W1 = sampled_weights[i, 0:(args.input_dim * args.H1)].reshape(args.input_dim, args.H1)
            temp = args.input_dim * args.H1
            W2 = sampled_weights[i, temp:(temp+args.H1*args.H2)].reshape(args.H1, args.H2)
            temp = temp+args.H1*args.H2
            W3 = sampled_weights[i, temp:(temp+args.H2*args.output_dim)].reshape(args.H2, args.output_dim)
            temp = temp+args.H2*args.output_dim
            B1 = sampled_weights[i,temp:(temp+args.H1)].reshape(1,args.H1)
            temp = temp+args.H1
            B2 = sampled_weights[i,temp:(temp+args.H2)].reshape(1,args.H2)
            temp = temp+args.H2
            B3 = sampled_weights[i,temp:(temp+args.output_dim)].reshape(1,args.output_dim)

            list_of_param_dicts.append({'W1': W1, 'W2': W2, 'W3': W3, 'B1': B1, 'B2': B2, 'B3': B3}.copy())

    return list_of_param_dicts


def calculate_nllsum_paramdict(args, y, x, param_dictionary):
    '''
    Given dataset name and model parameters w, return nL_n(w) = - \sum_{i=1}^n \log p(y|x,w)
    :return:
    '''

    if args.dataset == 'logistic_synthetic':

        prob = torch.sigmoid(torch.mm(x, param_dictionary['w']) + param_dictionary['b'])
        loss = nn.BCELoss(reduction='sum')

        return loss(prob, y)  # same as -sum(y*np.log(prob)+(1-y)*np.log(1-prob))

    elif args.dataset == 'tanh_synthetic':

        loss = nn.MSELoss(reduction='sum')
        mean = torch.matmul(torch.tanh(torch.matmul(x, param_dictionary['a'])), param_dictionary['b'])

        return len(y) * args.output_dim * 0.5 * np.log(2 * np.pi) + 0.5 * loss(y, mean)

    elif args.dataset == 'reducedrank_synthetic':

        loss = nn.MSELoss(reduction='sum')
        mean = torch.matmul(torch.matmul(x, param_dictionary['a']), param_dictionary['b'])

        return len(y) * args.output_dim * 0.5 * np.log(2 * np.pi) + 0.5 * loss(y, mean)

    elif args.dataset == 'ffrelu_synthetic':

        # calculate hidden and output layers
        h1 = F.relu((x @ param_dictionary['W1']) + param_dictionary['B1'])
        h2 = F.relu((h1 @ param_dictionary['W2']) + param_dictionary['B2'])
        mean = (h2 @ param_dictionary['W3']) + param_dictionary['B3']

        loss = nn.MSELoss(reduction='sum')

        return len(y) * args.output_dim * 0.5 * np.log(2 * np.pi) + 0.5 * loss(y, mean)



# TODO: this test module is from pyvarinf package, probably doesn't make sense for current framework
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


# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=7, verbose=False, delta=0):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement.
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#
#     def __call__(self, val_loss, model):
#
#         score = -val_loss
#
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0
#
#     def save_checkpoint(self, val_loss, model):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), 'checkpoint.pt')
#         self.val_loss_min = val_loss
