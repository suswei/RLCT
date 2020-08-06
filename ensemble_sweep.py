import sys
import os
import itertools

def main(taskid):

    MCs = 10

    taskid = int(taskid[0])
    hyperparameter_config = {
        'n': [500, 1000],
        'batchsize': [50,100],
        'output-dim': [5, 10],
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
              "--n %s "
              "--batchsize %s "
              "--output-dim %s "
              "--MC %s "
              %(taskid, temp['n'], temp['batchsize'], temp['output-dim'], temp['MC']))

if __name__ == "__main__":
    main(sys.argv[1:])
