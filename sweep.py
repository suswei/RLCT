import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])
    hyperparameter_config = {
        'syntheticsamplesize': [100, 200],
        'MCs': 5*[1]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    os.system("python3 main.py "
              "--taskid %s "
              "--w_dim 900 "
              "--syntheticsamplesize %s "
              "--dataset tanh_synthetic "
              "--sanity_check "
              "--betalogscale "
              "--numbetas 5 "
              "--MCs %s "
              "--num-warmup 10000 "
              "--num-samples 10000"
              %(taskid, temp['syntheticsamplesize'], temp['MCs']))

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
