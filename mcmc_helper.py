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