import sys
import os
import itertools

def main(taskid):

    experiment_name = 'submission'

    taskid = int(taskid[0])
    hyperparameter_config = {
        'X-test-std': [1.0, 3.0],
        'realizable': [0, 1],
        'minibatch': [0, 1],
        'rr-relu': [0, 1],
        'ffrelu-layers': [1, 5],
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    os.system("python3 lastlayerbayesian.py "
              "--num-n 10 --MCs 30 "
              "--experiment-name %s "
              "--taskid %s "
              "--X-test-std %s "
              "--realizable %s "
              "--rr-relu %s "
              "--ffrelu-layers %s "
              "--minibatch %s "
              %(experiment_name,
                taskid,
                temp['X-test-std'],
                temp['realizable'],
                temp['rr-relu'],
                temp['ffrelu-layers'],
                temp['minibatch']))


if __name__ == "__main__":
    main(sys.argv[1:])

