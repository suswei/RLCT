import sys
import os
import itertools

def main(taskid):

    experiment_name = 'small_w_dim'

    taskid = int(taskid[0])
    hyperparameter_config = {
        'X-test-std': [1.0],
        'realizable': [0, 1],
        'early-stopping': [0],
        'minibatch': [0],
        'input-dim': [3],
        'output-dim': [3],
        'rr-hidden': [3],
        'ffrelu-hidden': [5, 10],
        'ffrelu-layers': [1, 5],
        'mcmc_prior_map': [0]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    os.system("python3 lastlayerbayesian.py "
              "--num-n 5 --MCs 10 "
              "--experiment-name %s "
              "--taskid %s "
              "--X-test-std %s "
              "--realizable %s "
              "--early-stopping %s "
              "--input-dim %s "
              "--output-dim %s "
              "--rr-hidden %s "
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
                temp['ffrelu-hidden'],
                temp['ffrelu-layers'],
                temp['minibatch'],
                temp['mcmc_prior_map']))


if __name__ == "__main__":
    main(sys.argv[1:])

