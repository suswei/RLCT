import torch
import argparse
import time
import numpy as np
import pickle
import os

import pyro
import pyro.distributions as dist
from pyro.infer import HMC, MCMC, NUTS


# corresponds to pyro_tanh in models.py
def expected_nll_posterior_tanh(samples, args, X, Y):

    nll = []
    for r in range(args.num_samples):
        w = samples['w'][r]
        z1 = torch.tanh(torch.matmul(X, w))  # N D_H  <= first layer of activations
        q = samples['q'][r]
        z2 = torch.matmul(z1, q) # N D_H  <= second layer of activations
        ydist = dist.Normal(z2, 1)
        nll += [-ydist.log_prob(Y).sum()]

    nll_mean = sum(nll)/args.num_samples
    nll_var = sum((x - nll_mean) ** 2 for x in nll) / len(nll)

    return nll_mean, nll_var, nll

def expected_nll_posterior_relu(samples, args, X, Y):

    nll = []
    for r in range(args.num_samples):
        w1 = samples['w1'][r]
        b1 = samples['b1'][r]
        z1 = torch.relu(torch.matmul(X, w1) + b1)  # N D_H  <= first layer of activations

        w2 = samples['w2'][r]
        b2 = samples['b2'][r]
        z2 = torch.relu(torch.matmul(z1, w2) + b2)  # N D_H  <= first layer of activations

        w3 = samples['w3'][r]
        b3 = samples['b3'][r]
        z3 = torch.relu(torch.matmul(z2, w3) + b3)  # N D_H  <= first layer of activations

        w4 = samples['w4'][r]
        b4 = samples['b4'][r]
        z4 = torch.relu(torch.matmul(z3, w4) + b4)  # N D_H  <= first layer of activations

        w5 = samples['w5'][r]
        b5 = samples['b5'][r]
        z5 = torch.matmul(z4, w5) + b5  # N D_H  <= first layer of activations

        ydist = dist.Normal(z5, 1)
        nll += [-ydist.log_prob(Y).sum()]

    nll_mean = sum(nll)/args.num_samples
    nll_var = sum((x - nll_mean) ** 2 for x in nll) / len(nll)

    return nll_mean, nll_var, nll