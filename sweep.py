import sys
import os
import itertools

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
