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

def model_symmetric(X, Y, D_H, beta):

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
    D_H = args.symmetry_factor

    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=args.num_samples, warmup_steps=args.num_warmup)
    mcmc.run(X, Y, D_H, beta)
    print('\nMCMC elapsed time:', time.time() - start)
    return mcmc.get_samples()

# get data from symmetric model
def get_data_symmetric(args):
    D_X = 2
    N = args.num_data

    a = 2 * np.pi / args.symmetry_factor
    t1 = np.array([[np.cos(a / 2), np.sin(a / 2)]])

    w = np.vstack([np.matmul(t1, np.array([[np.cos(k * a), -np.sin(k * a)],
                                           [np.sin(k * a), np.cos(k * a)]]), dtype=np.float32) for k in range(args.symmetry_factor)])
    w = np.transpose(w)
    b = -0.3 * np.ones((args.symmetry_factor))
    q = np.ones((args.symmetry_factor, 1))
    c = np.array([0.0])

    w = torch.from_numpy(w)
    b = torch.from_numpy(b)
    q = torch.from_numpy(q)
    c = torch.from_numpy(c)

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

        # symmetric model
        w = samples['w'][r]
        b = samples['b'][r]
        z1 = nonlin(torch.matmul(X, w) + b)  # N D_H  <= first layer of activations
        q = samples['q'][r]
        c = samples['c'][r]
        z2 = nonlin(torch.matmul(z1, q) + c)  # N D_H  <= second layer of activations
        ydist = dist.Normal(z2, 1)
        nll += [-ydist.log_prob(Y).sum()]

    return (sum(nll)/args.num_samples)

def main(args):

    X, Y = get_data_symmetric(args)

    beta1 = 1.0/np.log(args.num_samples)
    beta2 = 1.5/np.log(args.num_samples)

    # do inference
    samples_beta1 = run_inference(model_symmetric, args, X, Y, beta=beta1)
    samples_beta2 = run_inference(model_symmetric, args, X, Y, beta=beta2)

    return (expected_nll_posterior(samples_beta1, X, Y) - expected_nll_posterior(samples_beta2, X, Y))/(1/beta1 - 1/beta2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian neural network example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=1000, type=int)
    parser.add_argument("--num-data", nargs='?', default=100, type=int)
    parser.add_argument("--symmetry-factor", nargs='?', default=3, type=int)
    args = parser.parse_args()

    path = './symm{}'.format(args.symmetry_factor)
    if not os.path.exists(path):
        os.makedirs(path)

    args_dict = vars(args)
    print(args_dict)
    with open('{}/config.pkl'.format(path), 'wb') as f:
        pickle.dump(args_dict, f)

    rlct = main(args)
    torch.save(rlct, '{}/rlct.pt'.format(path))
