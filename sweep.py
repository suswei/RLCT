import sys
import os
import itertools

def main(taskid):

    experiment_name = 'mcmc_last'

    taskid = int(taskid[0])
    hyperparameter_config = {
        'X-test-std': [1.0, 3.0],
        'realizable': [0, 1],
        'early-stopping': [0, 1],
        'minibatch': [0, 1],
        'input-dim': [3, 10],
        'output-dim': [3],
        'rr-hidden': [3],
        'rr-relu': [0, 1],
        'ffrelu-hidden': [5],
        'ffrelu-layers': [1, 5],
        'mcmc-prior-map': [0]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    os.system("python3 lastlayerbayesian.py "
              "--num-n 10 --MCs 20 "
              "--experiment-name %s "
              "--taskid %s "
              "--X-test-std %s "
              "--realizable %s "
              "--early-stopping %s "
              "--input-dim %s "
              "--output-dim %s "
              "--rr-hidden %s "
              "--rr-relu %s "
              "--ffrelu-hidden %s "
              "--ffrelu-layers %s "
              "--minibatch %s "
              "--mcmc-prior-map %s"
              %(experiment_name,
                taskid,
                temp['X-test-std'],
                temp['realizable'],
                temp['early-stopping'],
                temp['input-dim'],
                temp['output-dim'],
                temp['rr-hidden'],
                temp['rr-relu'],
                temp['ffrelu-hidden'],
                temp['ffrelu-layers'],
                temp['minibatch'],
                temp['mcmc-prior-map']))


if __name__ == "__main__":
    main(sys.argv[1:])

