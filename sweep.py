import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])
    hyperparameter_config = {
        'X-test-std': [1.0, 3.0],
        'realizable': [0, 1],
        'early-stopping': [0, 1],
        'input-dim': [3, 10],
        'output-dim': [3, 10],
        'rr-hidden': [3, 10],
        'ffrelu-hidden': [5, 10],
        'ffrelu-layers': [2, 10],
        'minibatch': [0, 1]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    os.system("python3 lastlayerbayesian.py "
              "--num-n 3 --MCs 2 --num-warmup 10 --R 100 "
              "--taskid %s "
              "--X-test-std %s "
              "--realizable %s "
              "--early-stopping %s "
              "--input-dim %s "
              "--output-dim %s "
              "--rr-hidden %s "
              "--ffrelu-hidden %s "
              "--ffrelu-layers %s "
              "--minibatch %s"
              %(taskid,
                temp['X-test-std'],
                temp['realizable'],
                temp['early-stopping'],
                temp['input-dim'],
                temp['output-dim'],
                temp['rr-hidden'],
                temp['ffrelu-hidden'],
                temp['ffrelu-layers'],
                temp['minibatch']))


if __name__ == "__main__":
    main(sys.argv[1:])

