from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.tools import add_constant
from scipy.linalg import toeplitz
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet
from matplotlib import pyplot as plt

import models


def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def retrieve_model(args):
    #M is the input dimension, H is hidden unit number, N is the output dimension for 'tanh_synthetic' and 'reducedrank_synthetic'
    # retrieve model
    if args.network == 'CNN':
        model = models.CNN(output_dim=args.output_dim)
        print('Error: implicit VI currently only supports logistic regression')
    if args.network == 'logistic':
        model = models.LogisticRegression(input_dim=args.input_dim, output_dim=args.output_dim)
    if args.network == 'FFrelu':
        model = models.FFrelu(input_dim=args.input_dim, output_dim=args.output_dim)
        print('Error: implicit VI currently only supports logistic regression')
    if args.network == 'tanh':
        model = models.tanh(input_dim=args.input_dim, output_dim=args.output_dim, H=args.H)
    if args.network == 'reducedrank':
        model = models.reducedrank(input_dim=args.input_dim, output_dim=args.output_dim, H=args.H)

    # TODO: count parameters automatically
    if args.network == 'logistic':
        if args.dataset in ('MNIST-binary', 'iris-binary', 'breastcancer-binary', 'lr_synthetic'):
            w_dim = (args.input_dim + 1)
        elif args.dataset == 'MNIST':
            w_dim = (args.input_dim + 1) * 9 / 2
    elif args.network in ['tanh', 'reducedrank']:
        w_dim = (args.input_dim + args.output_dim)*args.H
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


def lsfit_lambda(temperedNLL_perMC_perBeta, args, saveimgpath):

    # robust ls fit

    regr = ElasticNet(random_state=0,fit_intercept=True)
    regr.fit((1 / args.betas).reshape(args.numbetas,1), temperedNLL_perMC_perBeta)
    robust_intercept_estimate = regr.intercept_
    # slope_estimate = min(regr.coef_[0],args.w_dim/2)
    robust_slope_estimate = regr.coef_[0]

    # vanilla ols fit

    ols_model = OLS(temperedNLL_perMC_perBeta, add_constant(1 / args.betas)).fit()
    ols_intercept_estimate = ols_model.params[0]
    # slope_estimate = min(ols_model.params[1],args.w_dim/2)
    ols_slope_estimate = ols_model.params[1]


    plt.scatter(1 / args.betas, temperedNLL_perMC_perBeta, label='nll beta')
    plt.plot(1 / args.betas, robust_intercept_estimate + robust_slope_estimate * 1 / args.betas, 'g-', label='robust ols')
    plt.plot(1 / args.betas, ols_intercept_estimate + ols_slope_estimate * 1 / args.betas, 'b-', label='vanilla ols')

    plt.title("Thm 4, one MC realisation: d_on_2 = {}, true lambda = {:.1f} "
              "\n hat lambda robust = {:.1f}, hat lambda vanilla = {:.1f}"
              .format(args.w_dim/2, args.trueRLCT, robust_slope_estimate, ols_slope_estimate), fontsize=8)
    plt.xlabel("1/beta", fontsize=8)
    plt.ylabel("{} VI estimate of E^beta_w [nL_n(w)]".format(args.VItype), fontsize=8)
    plt.legend()
    plt.savefig('{}/thm4_beta_vs_lhs.png'.format(saveimgpath))
    plt.close()

    return robust_slope_estimate, ols_slope_estimate

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
