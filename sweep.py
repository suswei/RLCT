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

    taskid = int(taskid[0])
    hyperparameter_config = {
        # 'dataset': ['rr', 'lr', 'tanh', 'tanh_nontrivial'],
        'input_dim': [5],
        'output_dim': [5],
        'H0': [3],
        'H': [4],
        'alpha': [1.1,1.6,2.0]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    os.system("python3 langevin_monte_carlo.py "
              "--taskid %s "
              "--dataset rr "
              "--rr %s %s %s "
              "--H %s "
              "--method simsekli "
              "--simsekli-params %s %s %s"
              %(taskid, temp['input_dim'], temp['output_dim'], temp['H0'], temp['H'],1.1,0.5,2.0))

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

