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


def retrieve_model(args):
    #M is the input dimension, H is hidden unit number, N is the output dimension for '3layertanh_synthetic' and 'reducedrank_synthetic'
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

    # TODO: count parameters automatically
    if args.network == 'logistic':
        if args.dataset in ('MNIST-binary', 'iris-binary', 'breastcancer-binary', 'lr_synthetic'):
            w_dim = (args.input_dim + 1)
        elif args.dataset == 'MNIST':
            w_dim = (args.input_dim + 1) * 9 / 2
    elif args.network in ['Tanh', 'ReducedRankRegression']:
        w_dim = (args.input_dim + args.output_dim + 2)*args.H
    else:
        w_dim = count_parameters(model) * (args.output_dim - 1) / args.output_dim

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

    if np.all(np.linalg.eigvals(sigma) > 0):
        gls_model = GLS(temperedNLL_perMC_perBeta, add_constant(1 / betas), sigma=sigma).fit()
        return ols_model.params[1], gls_model.params[1]
    else:
        return ols_model.params[1], np.nan


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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
