from __future__ import print_function
import argparse
import pyvarinf
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from joblib import Parallel, delayed
import os
import random

import models
from dataset_factory import get_dataset_by_id


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(epoch,var_model,optimizer,weight):
    var_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.network == 'CNN':
            data, target = Variable(data), Variable(target)
        else:
            data, target = Variable(data.view(-1, 28 * 28)), Variable(target)
        optimizer.zero_grad()
        output = var_model(data) # what does var_model(data) actually do? does it first draw a sample of the network parameter and then apply the model?
        loss_error = F.nll_loss(output, target, weight=weight)
        loss_prior = var_model.prior_loss() / n  # TODO: why this division? documentation says "The model is only sent once, thus the division by the number of datapoints used to train"
        loss = loss_error + loss_prior  # this is the ELBO
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss error: {:.6f}\tLoss weights: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), loss_error.data.item(), loss_prior.data.item()))


def test(epoch,var_model):
    var_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            if args.network == 'CNN':
                data, target = Variable(data), Variable(target)
            else:
                data, target = Variable(data.view(-1, 28 * 28)), Variable(target)
            output = var_model(data)
            test_loss += F.nll_loss(output, target).data.item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def rsamples_nll(r,sample):
    nll = np.empty(0)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.network == 'CNN':
            data, target = Variable(data), Variable(target)
        else:
            data, target = Variable(data.view(-1, 28 * 28)), Variable(target)
        sample.draw()
        output = sample(data)
        nll = np.append(nll, np.array(F.nll_loss(output, target, reduction="sum").detach().cpu().numpy()))
    return nll.sum()


# TODO: this is only for categorical prediction at the moment, relies on nll_loss in pytorch. Should generalise eventually
def estimateBayesRLCT():

    # retrieve model
    if args.network == 'CNN':
        model = models.CNN(output_dim=output_dim)
    if args.network == 'logistic':
        model = models.LogisticRegression(input_dim=input_dim, output_dim=output_dim)
    if args.network == 'FFrelu':
        model = models.FFrelu(input_dim=input_dim, output_dim=output_dim)
    print('(number of trainable network parameters)/2: {}'.format(count_parameters(model) // 2))

    # variationalize model
    var_model = pyvarinf.Variationalize(model)
    var_model.set_prior(args.prior, **prior_parameters)
    if args.cuda:
        var_model.cuda()

    optimizer = optim.Adam(var_model.parameters(), lr=args.lr)

    device = torch.device("cuda" if args.cuda else "cpu")
    beta2 = 2.25/np.log(n) # if beta2 is not chosen carefully, we can easily get NaN for RLCT_estimate
    beta1 = 2/np.log(n)
    weight = beta1 * np.log(n) * torch.ones(output_dim) / np.log(args.batch_size)
    if args.cuda:
        weight = weight.to(device)

    # train
    for epoch in range(1, args.epochs + 1):
        train(epoch, var_model, optimizer, weight)
        test(epoch,var_model)

    # sample from variational distribution r
    sample = pyvarinf.Sample(var_model=var_model)


    my_list = range(args.R)
    num_cores = 1 # multiprocessing.cpu_count()
    nlls = Parallel(n_jobs=num_cores, verbose=50)(delayed(
        rsamples_nll)(i,sample) for i in my_list)
    #nlls = [rsamples_nll(i,sample) for i in my_list]
    nlls = np.asarray(nlls)

    RLCT_estimate = (nlls.mean() - (nlls*np.exp(-(beta2-beta1)*nlls)).mean()/(np.exp(-(beta2-beta1)*nlls)).mean())/(1/beta1-1/beta2)
    print('RLCT estimate: {}'.format(RLCT_estimate))

    if os.path.isfile('./{}.npy'.format(args.prior)):
        results = np.load('./{}.npy'.format(args.prior))
    else:
        results = np.empty(0)
    results = np.append(results,RLCT_estimate)
    np.save('./{}'.format(args.prior), results)


if __name__ == "__main__":

    random.seed()

    # Training settings
    parser = argparse.ArgumentParser(description='RLCT MNIST Example')
    parser.add_argument('--R', type=int, default=100,
                        help='number of MC draws from approximate posterior q (default:100')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--dataset-name', type=str, default='MNIST',help='dataset name from dataset_factory')
    parser.add_argument('--network',type=str, default='CNN', help='name of network in models')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--prior', type=str, default='gaussian', metavar='P',
                        help='prior used (default: gaussian)',
                        choices=['gaussian', 'mixtgauss', 'conjugate', 'conjugate_known_mean'])
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("args.cuda is " + str(args.cuda))

    # setting up prior parameters
    prior_parameters = {}
    if args.prior != 'gaussian':
        prior_parameters['n_mc_samples'] = 1
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

    # Daniel
    #torch.manual_seed(args.seed)
    #if args.cuda:
    #    torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader, test_loader, input_dim, output_dim = get_dataset_by_id(args,kwargs)

    n = len(train_loader.dataset)
    estimateBayesRLCT()

# can run this in bash
# for i in {1..200}; do python RLCT.py --R 500 --epochs 100; done
# saves the 200 MC results to "prior.npy"
