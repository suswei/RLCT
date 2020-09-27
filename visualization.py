import torch
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import argparse
import random
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from random import randint
import scipy.stats as st
import pickle
import math
import logging
import sys
from RLCT_helper import *

# TODO: Add 3D support
def posterior_viz(train_loader, sampled_weights, args, beta_index, saveimgpath):

    wmin = -3
    wmax = 3
    wspace = 50
    wrange = np.linspace(wmin, wmax, wspace)
    w = np.repeat(wrange[:, None], wspace, axis=1)
    w = np.concatenate([[w.flatten()], [w.T.flatten()]])

    # prior
    logprior = -(w ** 2).sum(axis=0) / 2

    # true unnormalised posterior at inverse temp
    logpost = torch.zeros(wspace * wspace)
    for i in range(wspace * wspace):
        current_w = torch.from_numpy(w[:,i]).float().unsqueeze(dim=0) # convert to torch tensor of shape [1,w_dim]
        param_dict = weights_to_dict(args, current_w)[0]
        logpost[i] = -args.betas[beta_index] * calculate_nllsum_paramdict(args, train_loader.dataset[:][1], train_loader.dataset[:][0], param_dict)
    logpost = logpost.detach().numpy() + logprior

    # kde sampled_weights
    kernel = st.gaussian_kde([sampled_weights[:,0].detach().numpy(), sampled_weights[:,1].detach().numpy()])
    f = np.reshape(kernel(w).T, [wspace,wspace])

    fig = make_subplots(rows=1, cols=3, subplot_titles=('(unnormalised) prior',
                                                        '(unnormalised) posterior',
                                                        'kde of sampled weights'))
    fig.add_trace(
        go.Contour(
            z=np.exp(logprior.reshape(wspace, wspace)),
            x=wrange,  # horizontal axis
            y=wrange
        ),  # vertical axis
        row=1, col=1
    )
    fig.add_trace(
        go.Contour(
            z=np.exp(logpost.reshape(wspace, wspace).T),
            x=wrange,  # horizontal axis
            y=wrange
        ),  # vertical axis
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=sampled_weights[:, 0].detach().numpy(),
            y=sampled_weights[:, 1].detach().numpy(),
            mode='markers',
            name='Sampled Data',
            marker=dict(size=3,opacity=0.6)
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Contour(
            z=f.T,
            x=wrange,  # horizontal axis
            y=wrange),  # vertical axis
            row=1, col=3
    )
    fig.update_layout(title_text='{}, {}, beta {}'.format(args.dataset, args.posterior_method, args.betas[beta_index]))
    if args.notebook:
        fig.show(renderer='notebook')
    else:
        fig.show()
    if saveimgpath is not None:
        fig.write_image('./{}/posteior_betaind{}'.format(saveimgpath, beta_index))


def tsne_viz(sampled_weights,args,beta_index,saveimgpath):

    if args.dataset == 'logistic_synthetic':
        true_weight = torch.cat((args.w_0.reshape(1, args.input_dim), args.b.reshape(1, 1)), 1)
    elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
        true_weight = torch.cat((args.a_params.reshape(1, (args.a_params.shape[0] * args.a_params.shape[1])), args.b_params.reshape(1, (args.b_params.shape[0] * args.b_params.shape[1]))), 1)

    sampled_true_w = torch.cat((sampled_weights, true_weight), 0)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(sampled_true_w.detach().numpy())
    tsne_results = pd.DataFrame(tsne_results, columns=['dim%s' % (dim_index) for dim_index in [1, 2]])
    tsne_results = pd.concat(
        [tsne_results, pd.DataFrame.from_dict({'sampled_true': ['sampled'] * (tsne_results.shape[0] - 1) + ['true']})],
        axis=1)
    plt.figure(figsize=(16, 10))
    ax = sns.scatterplot(x="dim1", y="dim2", hue="sampled_true", data=tsne_results)
    plt.title('tsne view: sampled w: beta = {}'.format(args.betas[beta_index]), fontsize=20)
    if saveimgpath is not None:
        plt.savefig('./{}/tsne_betaind{}'.format(saveimgpath, beta_index))
    plt.close()
