import sys
import os
import itertools
#
# def main(taskid):
#
#     taskid = int(taskid[0])
#
#     MCs = 5
#     hyperparameter_config = {
#         'dataset': ['rr'],
#         'n': [1000],
#         'batchsize': [100, 50],
#         'prior-std': [150.0, 1.0],
#         'y-std': [0.1, 1.0],
#         'MC': list(range(MCs))
#     }
#     keys, values = zip(*hyperparameter_config.items())
#     hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
#     temp = hyperparameter_experiments[taskid]
#
#     taskid = taskid // MCs
#
#     os.system("python3 ensembling_sgd.py "
#               # "--epochs 50 "
#               # "--numbetas 2 "
#               # "--R 5 "
#               "--taskid %s "
#               "--dataset %s "
#               "--n %s "
#               "--batchsize %s "
#               "--prior-std %s "
#               "--y-std %s "
#               "--MC %s "
#               % (taskid, temp['dataset'], temp['n'], temp['batchsize'], temp['prior-std'], temp['y-std'], temp['MC']))
#

def main(taskid):
    # TODO: some  combintations lead to clearly bad outcomes
    taskid = int(taskid[0])
    hyperparameter_config = {
        # 'dataset': ['rr', 'lr', 'tanh', 'tanh_nontrivial'],
        'input_dim': [10, 20],
        'output_dim': [10, 20],
        'H0': [3, 5, 10],
        'H': [10, 20],
        'alpha': [1.3, 1.5, 1.7, 2.0],
        'gamma': [0.5, 0.8, 1.0],
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    os.system("python3 langevin_monte_carlo.py "
              # "--R 2 --numbetas 2 "
              "--taskid %s "
              "--dataset rr "
              "--rr %s %s %s "
              "--method simsekli "
              "--simsekli-params %s %s"
              %(taskid, temp['input_dim'],temp['output_dim'],temp['H0'],temp['alpha'], temp['gamma']))

# def main(taskid):
#
#     taskid = int(taskid[0])
#     hyperparameter_config = {
#         'symmetry_factor': list(range(3, 11)),# 3 to 10
#         'num_hidden': [10, 15, 20],
#         'syntheticsamplesize': [100,250,500],
#         'mc': list(range(10))
#     }
#     keys, values = zip(*hyperparameter_config.items())
#     hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
#     temp = hyperparameter_experiments[taskid]
#
#     os.system("python3 pyro_example.py --symmetry-factor %s --num-hidden %s --num-data %s --mc %s"
#               %(temp['symmetry_factor'],temp['num_hidden'],temp['syntheticsamplesize'],temp['mc']))


if __name__ == "__main__":
    main(sys.argv[1:])
