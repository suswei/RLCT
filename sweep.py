import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])
    hyperparameter_config = {
        'dataset': ['reducedrank_synthetic', 'tanh_synthetic'],
        'syntheticsamplesize': [500, 1000, 5000],
        'n_hidden_D': [128, 256],
        'num_hidden_layers_D': [1, 2],
        'n_hidden_G': [128, 256],
        'num_hidden_layers_G': [1, 2],
        'MCs': 50*[1]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    os.system("python3 main.py --sanity_check --dpower 0.4 --VItype implicit --epochs 200 --MCs 1 --betasbegin 0.1 --betasend 0.5 --betalogscale --numbetas 10 "
              "--taskid %s --dataset %s --syntheticsamplesize %s "
              "--n_hidden_D %s --num_hidden_layers_D %s --n_hidden_G %s --num_hidden_layers_G %s"
              %(taskid, temp['dataset'], temp['syntheticsamplesize'],
                temp['n_hidden_D'], temp['num_hidden_layers_D'], temp['n_hidden_G'], temp['num_hidden_layers_G'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])
