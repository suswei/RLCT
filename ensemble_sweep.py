import sys
import os
import itertools


def main(taskid):

    taskid = int(taskid[0])

    MCs = 5
    hyperparameter_config = {
        'dataset': ['rr'],
        'n': [1000],
        'batchsize': [100, 50],
        'prior-std': [150.0, 1.0],
        'y-std': [0.1, 1.0],
        'MC': list(range(MCs))
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    taskid = taskid // MCs

    os.system("python3 ensembling_main.py "
              # "--epochs 50 "
              # "--numbetas 2 "
              # "--R 5 "
              "--taskid %s "
              "--dataset %s "
              "--n %s "
              "--batchsize %s "
              "--prior-std %s "
              "--y-std %s "
              "--MC %s "
              % (taskid, temp['dataset'], temp['n'], temp['batchsize'], temp['prior-std'], temp['y-std'], temp['MC']))


if __name__ == "__main__":
    main(sys.argv[1:])

