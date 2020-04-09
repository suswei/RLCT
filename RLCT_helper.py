from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.tools import add_constant
from scipy.linalg import toeplitz

import models


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


# perform some data massaging to get right dimensions
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


# TODO: this test module was copied from original pyvarinf package, needs to be updated to fit into current framework
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