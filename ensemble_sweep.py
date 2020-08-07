import sys
import os
import itertools


def main(taskid):

    MCs = 5

    taskid = int(taskid[0])
    hyperparameter_config = {
        'dataset': ['rr','tanh'],
        'n': [500, 1000],
        'batchsize': [100, 500],
        'H': [5, 10],
        'MC': list(range(MCs))
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    taskid = taskid//MCs

    os.system("python3 ensembling_main.py "
              # "--epochs 50 "
              # "--numbetas 2 "
              # "--R 5 "
              "--taskid %s "
              "--dataset %s "
              "--n %s "
              "--batchsize %s "
              "--H %s "
              "--MC %s "
              %(taskid, temp['dataset'], temp['n'], temp['batchsize'], temp['H'], temp['MC']))


if __name__ == "__main__":
    main(sys.argv[1:])
