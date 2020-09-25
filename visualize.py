import torch
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import pandas as pd


def main(experiment_name, taskid):

    # load simulation resutls
    args = torch.load('lastlayersims/{}_taskid{}_args.pt'.format(experiment_name, taskid))
    n_range = args.n_range
    results = torch.load('lastlayersims/{}_taskid{}_results.pt'.format(experiment_name, taskid))

    # rename to prettier key names in results
    results['last two layers (A,B) MCMC'] = results.pop('mcmc_rr')
    results['last layer only (B) MCMC'] = results.pop('mcmc_last')
    results['last two layers (A,B) Laplace'] = results.pop('laplace_rr')
    results['last layer only (B) Laplace'] = results.pop('laplace_last')
    results['MAP'] = results.pop('map')
    del results['entropy']

    # get learning coefficient table
    avg_gen_err = dict()
    for key in results:
        avg_gen_err[key] = [i.mean() for i in results[key]]

    methods = []
    learning_coefficients = []
    R2 = []
    for k, v in avg_gen_err.items():
        if k != 'entropy':
            if args.realizable == 1:
                ols = OLS(v, 1 / n_range).fit()
                learning_coefficients += [ols.params[0]]
            else:
                ols = OLS(v, add_constant(1 / n_range)).fit()
                learning_coefficients += [ols.params[1]]

            methods += [k]
            R2 += [ols.rsquared]

    lambdas = pd.DataFrame({'method': methods, 'learning coefficient': learning_coefficients,'R squared': R2})
    print(lambdas.to_latex(index=False))

    with open('lastlayersims/publication_taskid{}.tex'.format(args.taskid), 'w') as tf:
        tf.write(lambdas.to_latex())

    # create list of pandas dataframes for plotting n versus E_n G(n)
    df = []
    for key in results:
        if key not in ['last two layers (A,B) Laplace', 'last layer only (B) Laplace']:
            for i in range(len(n_range)):
                df += [pd.DataFrame({'sample size': np.repeat(n_range[i], args.MCs),
                                     'average generalization error': results[key][i],
                                     'method': np.repeat(key, args.MCs)})]
    all_results = pd.concat(df)
    ax = sns.pointplot(x="sample size", y="average generalization error", hue="method",
                       data=all_results, dodge=True)
    title = ''
    title += '{} hidden layer(s) in g network, '.format(args.ffrelu_layers)
    if args.use_rr_relu:
        title += 'ReLU activation in h network'
    else:
        title += 'identity activation in h network'

    plt.title(title)
    plt.savefig('lastlayersims/publication_taskid{}.png'.format(args.taskid))
    plt.close()




experiment_name ='submission'
for taskid in range(32):
    main(experiment_name,taskid)