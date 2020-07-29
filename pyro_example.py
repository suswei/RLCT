# code is from this mish-mash
# http://pyro.ai/numpyro/bnn.html
# http://docs.pyro.ai/en/stable/mcmc.html

import torch
import argparse
import time
import numpy as np
import pickle
import os

import pyro
import pyro.distributions as dist
from pyro.infer import HMC, MCMC, NUTS


# the non-linearity we use in our neural network
def nonlin(x):
    return torch.relu(x)


# feedforward relu network with D_H hidden units, at "temperature" 1/beta
def model(X, Y, D_H, beta):
    D_X, D_Y = X.shape[1], 1

    # sample first layer (we put unit normal priors on all weights)
    w = pyro.sample("w", dist.Normal(torch.zeros((D_X, D_H)), torch.ones((D_X, D_H))))  # D_X D_H
    b = pyro.sample("b", dist.Normal(torch.zeros((1, D_H)), torch.ones((1, D_H))))  # D_X D_H
    z1 = nonlin(torch.matmul(X, w) + b)   # N D_H  <= first layer of activations

    # sample second layer
    q = pyro.sample("q", dist.Normal(torch.zeros((D_H, 1)), torch.ones((D_H, 1))))  # D_H D_H
    c = pyro.sample("c", dist.Normal(torch.zeros((1, 1)), torch.ones((1, 1))))  # D_H D_H
    z2 = nonlin(torch.matmul(z1, q)+c)  # N D_H  <= second layer of activations

    # observe data
    pyro.sample("Y", dist.Normal(z2, 1/np.sqrt(beta)), obs=Y)


# helper function for HMC inference
def run_inference(model, args, X, Y, beta):
    D_H = args.num_hidden

    start = time.time()
    kernel = NUTS(model, adapt_step_size=True)
    mcmc = MCMC(kernel, num_samples=args.num_samples, warmup_steps=args.num_warmup)
    mcmc.run(X, Y, D_H, beta)
    print('\nMCMC elapsed time:', time.time() - start)
    return mcmc


# get data from symmetric true distribution
def get_data_symmetric(args):

    D_X = 2
    N = args.num_data
    symmetry_factor = args.symmetry_factor
    num_hidden = args.num_hidden

    a = 2 * np.pi / args.symmetry_factor
    t1 = np.array([[np.cos(a / 2), np.sin(a / 2)]])

    # w = np.vstack([np.matmul(t1, np.array([[np.cos(k * a), -np.sin(k * a)],
    #                                        [np.sin(k * a), np.cos(k * a)]]), dtype=np.float32) for k in range(args.symmetry_factor)])
    # w = np.transpose(w)
    # b = -0.3 * np.ones((args.symmetry_factor))
    # q = np.ones((args.symmetry_factor, 1))
    # c = np.array([0.0])

    w_list = [np.matmul(t1, np.array([[np.cos(k * a), -np.sin(k * a)],
                                      [np.sin(k * a), np.cos(k * a)]])) for k in range(symmetry_factor)]
    w_list.extend([np.zeros_like(w_list[0]) for k in range(num_hidden - symmetry_factor)])
    w = np.vstack(w_list)

    w = np.transpose(w)
    b = np.concatenate([-0.3 * np.ones((symmetry_factor)), np.zeros((num_hidden - symmetry_factor))], axis=0)

    # q = np.transpose(np.vstack([np.ones((num_hidden)), np.zeros((num_hidden))]))
    q = np.concatenate([np.ones((symmetry_factor, 1)), np.zeros((num_hidden - symmetry_factor, 1))], axis=0)
    c = np.array([0.0])

    w = torch.from_numpy(w).float()
    b = torch.from_numpy(b).float()
    q = torch.from_numpy(q).float()
    c = torch.from_numpy(c).float()

    X = torch.linspace(-1, 1, N)
    X = torch.pow(X[:, np.newaxis], torch.arange(D_X))
    z1 = nonlin(torch.matmul(X, w) + b)  # N D_H  <= first layer of activations
    z2 = nonlin(torch.matmul(z1, q) + c)  # N D_H  <= second layer of activations

    sigma_obs = 1.0
    Y = z2 + sigma_obs * torch.randn(N)
    Y = Y[:, np.newaxis]

    return X, Y


def expected_nll_posterior(samples, X, Y):

    nll = []
    for r in range(args.num_samples):
        w = samples['w'][r]
        b = samples['b'][r]
        z1 = nonlin(torch.matmul(X, w) + b)  # N D_H  <= first layer of activations
        q = samples['q'][r]
        c = samples['c'][r]
        z2 = nonlin(torch.matmul(z1, q) + c)  # N D_H  <= second layer of activations
        ydist = dist.Normal(z2, 1)
        nll += [-ydist.log_prob(Y).sum()]

    return sum(nll)/args.num_samples


def main(args):
    path = args.path
    X, Y = get_data_symmetric(args)

    beta1 = 1.0/np.log(args.num_samples)
    beta2 = 1.5/np.log(args.num_samples)

    # do inference
    mcmc_beta1 = run_inference(model, args, X, Y, beta=beta1)
    mcmc_beta2 = run_inference(model, args, X, Y, beta=beta2)

    torch.save(mcmc_beta1, '{}/mcmc_beta1.pt'.format(path))
    torch.save(mcmc_beta2, '{}/mcmc_beta2.pt'.format(path))

    rlct = (expected_nll_posterior(mcmc_beta1.get_samples(), X, Y) - expected_nll_posterior(mcmc_beta2.get_samples(), X, Y))/(1/beta1 - 1/beta2)
    torch.save(rlct, '{}/rlct.pt'.format(path))
    print('rlct {}'.format(rlct))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLCT_HMC_symmetric")
    parser.add_argument("--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=1000, type=int)
    parser.add_argument("--num-data", nargs='?', default=100, type=int)
    parser.add_argument("--symmetry-factor", nargs='?', default=3, type=int)
    parser.add_argument("--num-hidden", nargs='?', default=10, type=int)
    parser.add_argument("--mc", default=1, type=int)

    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)

    # create path
    args.path = './symm{}_numhidden{}_numdata{}_mc{}'\
        .format(args.symmetry_factor,args.num_hidden, args.num_data, args.mc)
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # save simulation setting
    torch.save(args_dict, '{}/config.pt'.format(args.path))

    main(args)

