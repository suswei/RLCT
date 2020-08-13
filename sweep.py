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
        'sas_alpha': [1.3,1.5,1.7,1.9],
        'exp_schedule': [0.6,0.8,1.0],
        # 'stepsize_constant': [True,False],
        'eps': [1e-8, 1e-9]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    os.system("python3 langevin_monte_carlo.py "
              "--taskid %s --method simsekli --sas-alpha %s --exp-schedule %s"
              %(taskid, temp['sas_alpha'], temp['exp_schedule']))

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
