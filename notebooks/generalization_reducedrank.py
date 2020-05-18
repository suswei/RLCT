#!/usr/bin/env python
# coding: utf-8

# # Generalisation error
#
# In this experiment we vary the architecture of a feedforward ReLU network
# and plot the (average) Bayesian generalization error as a function over the architecture.
#
# We estimate the (average) Bayesian generalisation error as
# \begin{equation}
# E_n B_g(n) \approx E_n \frac{1}{n'} \sum_{i=1}^n' \log \frac{q(y_i|x_i)}{p(y_i |x_i,, \mathcal D_n)}
# \end{equation}
# where $\mathcal D_n$ is the training set and $n'$ is the size of the test set.
#
#

# In[14]:


from __future__ import print_function

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

import sys
import numpy as np

sys.path.append('../')
from main import *

class Args:

    dataset = 'reducedrank_synthetic'
    sanity_check = True
    syntheticsamplesize = 200

    network = 'reducedrank'
    VItype = 'implicit'
    batchsize = 100
    epochs = 400
    epsilon_mc = 100
    pretrainDepochs = 100
    trainDepochs = 50
    n_hidden_D = 128
    num_hidden_layers_D = 1
    n_hidden_G = 128
    num_hidden_layers_G = 1

    lr_primal = 1e-3
    lr_dual = 5e-2

    beta_auto_liberal = False
    beta_auto_conservative = False
    beta_auto_oracle = False
    betasbegin = 0.1
    betasend = 1.5
    betalogscale = True
    numbetas = 10

    elasticnet_alpha = 1.0
    R = 200

    MCs = 10

    log_interval = 50

    cuda = False

args = Args()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


# Making sure we satisfy the theoretical conditions, the true distribution is made to be realizable.
# In particular we generate data from with input and output dimension equal to 1 and $H_1=2, H_2=1$ from the following model
#
# ```python
# class ffrelu(nn.Module):
#     def __init__(self,input_dim, output_dim):
#         super(ffrelu, self).__init__()
#         self.fc1 = nn.Linear(input_dim, H1)
#         self.fc2 = nn.Linear(H1, H2)
#         self.fc3 = nn.Linear(H2, output_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
# ```

# In[15]:


args.input_dim = 6
args.output_dim = 6

args.a_params = torch.randn(args.input_dim,3)
args.b_params = torch.randn(3,args.output_dim)
H0 = torch.matrix_rank(torch.matmul(args.a_params,args.b_params))

# In[16]:


def train(args, train_loader, valid_loader):

    # get a grid of inverse temperatures [beta_1/log n, \ldots, beta_k/log n]
    set_betas(args)

    mc = 1
    saveimgpath = None
    nll_betas_implicit = np.empty(0)

    for beta_index in range(args.betas.shape[0]):

        # train implicit variational inference
        print('Begin training IVI')
        G = train_implicitVI(train_loader, valid_loader, args, mc, beta_index, saveimgpath)
        print('Finished training IVI')
        nllw_array_implicit = approxinf_nll_implicit(train_loader, G, args)
        nllw_mean_implicit = sum(nllw_array_implicit)/len(nllw_array_implicit)
        nll_betas_implicit = np.append(nll_betas_implicit, nllw_mean_implicit)

    ivi_robust, ivi_ols = lsfit_lambda(nll_betas_implicit, args, saveimgname=None)

    return ivi_robust

def compute_predictive_dist(args, G, loader):

    R = 1000
    eps = torch.randn(R, args.epsilon_dim)
    sampled_weights = G(eps)
    wholex = loader.dataset[:][0]
    wholey = loader.dataset[:][1]
    list_of_param_dicts = weights_to_dict(args, sampled_weights)

    pred_logprob = 0
    for param_dict in list_of_param_dicts:

        mean = torch.matmul(torch.matmul(wholex, param_dict['a']), param_dict['b'])
        pred_logprob += -(args.output_dim*np.log(2 * np.pi) + torch.norm(wholey-mean,dim=1)**2) / 2

    return pred_logprob/R


def compute_EBg(pred_logprob,loader,args):

    wholex = loader.dataset[:][0]
    wholey = loader.dataset[:][1]
    mean = torch.matmul(torch.matmul(wholex, args.a_params), args.b_params)
    mean_dev = torch.norm(wholey-mean,dim=1)**2
    true_logprob = -(args.output_dim * np.log(2 * np.pi) + mean_dev) / 2

    return (true_logprob-pred_logprob).mean()


# In[17]:

Hrange = range(3, 6)
results=[]

for H in Hrange:

        args.H = H
        args.model, args.w_dim = retrieve_model(args)
        args.epsilon_dim = args.w_dim

        Bg = np.empty(args.MCs)
        rlct = np.empty(args.MCs)

        for mc in range(0, args.MCs):

            train_loader, valid_loader, test_loader = get_dataset_by_id(args, kwargs)

            args.betas = [1.0]
            beta_index = 0
            G = train_implicitVI(train_loader, valid_loader, args, mc, beta_index, saveimgpath=None)
            with torch.no_grad():
                pred = compute_predictive_dist(args, G, test_loader)
                Bg[mc] = compute_EBg(pred, test_loader, args)

            rlct[mc] = train(args, train_loader, valid_loader)
            print('reduced rank regression model H = {}'.format(H))
            print('mc {}: Bg {}'.format(mc, Bg[mc]))
            print('mc {}: rlct {}'.format(mc, rlct[mc]))

        print('H: {}'.format(H))
        print('E_n Bg(n): {}'.format(Bg.mean()))
        print('hat RLCT/n: {}'.format(rlct.mean() / args.syntheticsamplesize))
        results += {'H':H,'E_n Bg(n)': Bg.mean(), 'hat RLCT/n': rlct.mean()/ args.syntheticsamplesize}
# In[ ]:

with open('generalization_results.pkl', 'wb') as f:
    pickle.dump(results, f)