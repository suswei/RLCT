#!/usr/bin/env python
# coding: utf-8

# # Generalisation error
# 
# In this experiment we vary the architecture of a feedforward ReLU network
# and examine the (average) Bayesian generalization error as a function over the architecture.
# 
# We estimate the (average) Bayesian generalisation error as
# \begin{equation}
# E_n B_g(n) \approx E_n \frac{1}{n'} \sum_{i=1}^{n'} \log \frac{q(y_i|x_i)}{p(y_i |x_i,, \mathcal D_n)}
# \end{equation}
# where $\mathcal D_n$ is the training set and $n'$ is the size of the test set.
# Note that the predictive distribution $p(y_i |x_i,, \mathcal D_n)$ is with respect to inverse temperature $\beta =1$
# 
# To begin, we set some global parameters for this experiment.

# In[ ]:


from __future__ import print_function
import sys
sys.path.append('../')
from main import *

class Args:

    dataset = 'ffrelu_synthetic'
    sanity_check = True
    syntheticsamplesize = 500

    network = 'ffrelu'
    VItype = 'implicit'
    batchsize = 100
    epochs = 200
    epsilon_mc = 100
    pretrainDepochs = 10
    trainDepochs = 2
    n_hidden_D = 50
    num_hidden_layers_D = 2
    n_hidden_G = 50
    num_hidden_layers_G = 2

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


# We design the true distribution to be realizable.
# In particular we generate univariate Gaussian (mean $0$ and variance $1$) input data $X$
# and set the univariate target $Y$ to be Gaussian with variance 1 and
# mean given by ```ffrelu```$(X,H_1=2,H_2=1)$ according to the following network
# 
# ```python
# class ffrelu(nn.Module):
#     def __init__(self,input_dim, output_dim,H1,H2):
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
# 
# Below are some helper functions:
# - ```train``` outputs an estimate of the RLCT based on a grid of inverse temperatures $\beta$
# - ```compute_predictive_dist``` takes in a generator trained on $D_n$ at $\beta = 1$ and computes $p(y_i|x_i,D_n)$ by sampling
# - ```compute_Bg``` uses the result of ```compute_predictive_dist``` and the same testing dataset to compute $B_g = \sum_{i=1}^{n'} \log \frac{q(y_i|x_i)}{p(y_i |x_i,, \mathcal D_n)}$

# In[ ]:


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

        h1 = F.relu((wholex @ param_dict['W1']) + param_dict['B1'])
        h2 = F.relu((h1 @ param_dict['W2']) + param_dict['B2'])
        mean = (h2 @ param_dict['W3']) + param_dict['B3']
        pred_logprob += -(np.log(2 * np.pi) + (wholey - mean) ** 2) / 2

    return pred_logprob/R


def compute_Bg(pred_logprob,loader,args):

    wholex = loader.dataset[:][0]
    wholey = loader.dataset[:][1]
    true_logprob = -(np.log(2 * np.pi) + (wholey - args.true_mean(wholex)) ** 2) / 2

    return (true_logprob-pred_logprob).mean()


# We range over the model's architecture as determined by $H_1$ and $H_2$.
# For each architecture we estimate the RLCT and calculate the average Bayesian generalization error.
# Hopefully we can see the relationship $E_n B_g(n) \approx \lambda/n$.

# In[ ]:


args.input_dim = 1
args.output_dim = 1
args.true_mean = models.ffrelu(args.input_dim, args.output_dim, 2, 1)

H1range = range(1,4)
H2range = range(1,3)
results = []

for H1 in H1range:
    for H2 in H2range:

        args.H1 = H1
        args.H2 = H2
        args.model, args.w_dim = retrieve_model(args)
        args.epsilon_dim = args.w_dim

        Bg = np.empty(args.MCs)
        rlct = np.empty(args.MCs)

        for mc in range(0,args.MCs):

            train_loader, valid_loader, test_loader = get_dataset_by_id(args, kwargs)

            args.betas = [1.0]
            beta_index = 0
            G = train_implicitVI(train_loader, valid_loader, args, mc, beta_index, saveimgpath=None)
            pred = compute_predictive_dist(args, G, test_loader)
            Bg[mc] = compute_Bg(pred,test_loader,args)

            rlct[mc] = train(args, train_loader, valid_loader)

        print('H1: {}, H2: {}'.format(H1,H2))
        print('E_n Bg(n): {}'.format(Bg.mean()))
        print('hat RLCT/n: {}'.format(rlct.mean()/args.syntheticsamplesize))
        results += {'H1':H1, 'H2':H2, 'E_n Bg(n)': Bg.mean(), 'hat RLCT/n': rlct.mean()/ args.syntheticsamplesize}


with open('generalization_ffrelu.pkl', 'wb') as f:
    pickle.dump(results, f)

