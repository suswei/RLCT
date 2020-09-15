# code is from this mish-mash
# http://pyro.ai/numpyro/bnn.html
# http://docs.pyro.ai/en/stable/mcmc.html

#
# TODO: 
# - parallelisation checks
# - different true distribution
# - different nonlinearity

import torch
import argparse
import time
import numpy as np
import pickle
import os
import math
import torch.multiprocessing
from torch.multiprocessing import Process, Manager

import pyro
import pyro.distributions as dist
from pyro.infer import HMC, MCMC, NUTS

from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# This is required both to get AMD CPUs to work well, but also
# to disable the aggressive multi-threading of the underlying
# linear algebra libraries, which interferes with our multiprocessing
# with PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_SERIAL'] = 'YES'
os.environ['OMP_NUM_THREADS'] = '1'

#export CUDA_VISIBLE_DEVICES=""
#export MKL_DEBUG_CPU_TYPE=5
#export MKL_SERIAL=YES; export OMP_NUM_THREADS=1

# the non-linearity we use in our neural network
def nonlin(x):
    return torch.nn.functional.relu(x)
    #return torch.tanh(x)

# feedforward relu network with H hidden units, at "temperature" 1/beta
# note that the return 
def model(X, Y, H, beta, prior_sd):
    M, N = X.shape[1], Y.shape[1]

    # w is the weight matrix R^2 --> R^H
    w = pyro.sample("w", dist.Normal(torch.zeros((M, H)), prior_sd * torch.ones((M, H)))) 
    
    # b is the bias in the hidden layer
    b = pyro.sample("b", dist.Normal(torch.zeros(H), prior_sd * torch.ones(H)))
    
    # q is the weight matrix R^H --> R^1
    q = pyro.sample("q", dist.Normal(torch.zeros((H, N)), prior_sd * torch.ones((H, N))))
    
    # c is the final bias
    c = pyro.sample("c", dist.Normal(torch.zeros(N), prior_sd * torch.ones(N)))
    
    a = torch.matmul(X, w) + b
    f = torch.matmul(nonlin(a), q) + c
     
    return pyro.sample("Y", dist.Normal(f, 1/np.sqrt(beta)), obs=Y)

# helper function for HMC inference
def run_inference(model, args, X, Y, beta, beta_num, samples):
    H = args.num_hidden

    start = time.time()
    kernel = NUTS(model, adapt_step_size=True, 
                target_accept_prob=args.target_accept_prob,
                jit_compile=args.jit)
    mcmc = MCMC(kernel, num_samples=args.num_samples, warmup_steps=args.num_warmup)
    mcmc.run(X, Y, H, beta, args.prior_sd)
    print("\n[beta = {}]".format(beta))
    mcmc.summary(prob=0.5)

    torch.save(mcmc.get_samples(), '{}/mcmc_beta{}_samples.pt'.format(args.path, beta_num))
    torch.save(time.time() - start, '{}/mcmc_beta{}_time_secs.pt'.format(args.path, beta_num))
    samples[beta_num] = mcmc.get_samples()


# get data from true distribution
#def get_data_true(args):
#    num_data = args.num_data
#    M = args.num_input_nodes
#    N = args.num_output_nodes

#    X = 2 * args.x_max * torch.rand(num_data, M) - args.x_max
#    Y = torch.randn(num_data, N)

#    return X, Y

def get_data_true(args):
    # Sample from q(x) in R^2
    num_data = args.num_data
    M = args.num_input_nodes
    N = args.num_output_nodes

    X = 2 * args.x_max * torch.rand(num_data, M) - args.x_max
    
    # Construct the symmetric true distribution
    angle = 2 * np.pi / args.num_hidden_true
    t1 = np.array([[np.cos(angle/2), np.sin(angle/2)]])

    # The true distribution uses the beginning segment of the hidden nodes to encode
    # the hyperplanes bounding the polygon, and puts zeros for all other weights
    w_list = [ np.matmul(t1, np.array([[np.cos(k*angle), -np.sin(k*angle)],
                                             [np.sin(k*angle), np.cos(k*angle)]])) for k in range(args.num_hidden_true)]
    w_list.extend([ np.zeros_like(w_list[0]) for k in range(args.num_hidden-args.num_hidden_true)])
    w = np.vstack(w_list)

    w = np.transpose(w)
    b = np.concatenate([-0.3 * np.ones((args.num_hidden_true)), np.zeros((args.num_hidden-args.num_hidden_true))],axis=0)

    #q = np.transpose(np.vstack([np.ones((num_hidden)), np.zeros((num_hidden))]))
    q = np.concatenate([np.ones((args.num_hidden_true,1)), np.zeros((args.num_hidden-args.num_hidden_true,1))],axis=0)
    c = np.array([0.0])

    w_t = torch.tensor(w, dtype=torch.float)
    b_t = torch.tensor(b, dtype=torch.float)
    q_t = torch.tensor(q, dtype=torch.float)
    c_t = torch.tensor(c, dtype=torch.float)
    
    a = torch.matmul(X, w_t) + b_t
    f = torch.matmul(nonlin(a), q_t) + c_t # has shape (
    
    
    ydist = dist.Normal(f, 1)
    Y = ydist.sample()
    
    return X, Y

def expected_nll_posterior(samples, X, Y):

    nll = []
    for r in range(args.num_samples):
        w = samples['w'][r]
        b = samples['b'][r]
        q = samples['q'][r]
        c = samples['c'][r]
        
        a = torch.matmul(X, w) + b
        f = torch.matmul(nonlin(a), q) + c
    
        ydist = dist.Normal(f, 1)
        nll += [-ydist.log_prob(Y).sum()]

    return sum(nll)/args.num_samples


def main(args):
    path = args.path
    n = args.num_data # fix 15-9-2020 was args.num_samples

    X, Y = get_data_true(args)

    betas = np.linspace(1 / np.log(n) * (1 - 1 / np.sqrt(2 * np.log(n))),
                        1 / np.log(n) * (1 + 1 / np.sqrt(2 * np.log(n))), args.num_betas)

    # do inference
    manager = Manager()
    samples = manager.dict()
    jobs = []
    for i in range(len(betas)):
        p = Process(target=run_inference, args=(model, args, X, Y, betas[i], i, samples))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

    # rlct_estimate = (expected_nll_posterior(samples[0], X, Y) - expected_nll_posterior(samples[1], X, Y))/(1/betas[0] - 1/betas[1])

    estimates = [expected_nll_posterior(samples[i], X, Y) for i in range(len(samples))]
    regr = LinearRegression(fit_intercept=True)
    one_on_betas = (1 / betas).reshape(args.num_betas, 1)
    regr.fit(one_on_betas, estimates)
    score = regr.score(one_on_betas, estimates)
    b_ols = regr.intercept_
    m_ols = regr.coef_[0]

    torch.save(m_ols, '{}/rlct_estimate.pt'.format(path))
    print('RLCT estimate {} with r2 coeff {}'.format(m_ols, score))

    #rlct_true = (math.floor(math.sqrt(H)) ** 2 + math.floor(math.sqrt(H)) + H) / (4 * math.sqrt(H) + 2)
    #torch.save(rlct_true, '{}/rlct_true.pt'.format(path))
    #print('rlct true {}'.format(rlct_true))

    plt.figure()
    plt.title("E^beta_w[nL_n(w)] against 1/beta for single dataset")
    plt.scatter(1/betas, np.array(estimates))
    plt.plot(1/betas, [m_ols * x + b_ols for x in 1/betas], label='ols')
    plt.legend(loc='best')
    plt.savefig("{}/linfit.png".format(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLCT_HMC_symmetric")
    parser.add_argument("--experiment-id", nargs="?")
    parser.add_argument("--save-prefix", nargs="?")
    parser.add_argument("--num-samples", nargs="?", default=100000, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=30000, type=int)
    parser.add_argument("--num-data", nargs='?', default=1000, type=int)
    # M
    parser.add_argument("--num-input-nodes", nargs='?', default=2, type=int)
    # N
    parser.add_argument("--num-output-nodes", nargs='?', default=1, type=int)
    # H
    parser.add_argument("--num-hidden", nargs='?', default=4, type=int)
    # H_0
    parser.add_argument("--num-hidden-true", nargs='?', default=0, type=int)
    parser.add_argument("--prior-sd", nargs='?', default=1.0, type=float)
    parser.add_argument("--x-max", nargs='?', default=1, type=int)
    parser.add_argument("--target-accept-prob", nargs='?', default=0.8, type=float)
    parser.add_argument("--num-betas", default=8, type=int)

    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)

    args_filename = args.save_prefix + '/' + args.experiment_id + '-args.pickle'
    
    # create path
    args.path = args.save_prefix + '/{}'.format(args.experiment_id)
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # save simulation setting
    torch.save(args, '{}/args.pt'.format(args.path))

    # for GPU see https://github.com/pyro-ppl/pyro/blob/dev/examples/baseball.py
    # work around the error "CUDA error: initialization error"
    # see https://github.com/pytorch/pytorch/issues/2517
    torch.multiprocessing.set_start_method("spawn")
    
    # work around with the error "RuntimeError: received 0 items of ancdata"
    # see https://discuss.pytorch.org/t/received-0-items-of-ancdata-pytorch-0-4-0/19823
    torch.multiprocessing.set_sharing_strategy("file_system")
        
    main(args)