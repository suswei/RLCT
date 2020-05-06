import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])
    hyperparameter_config = {
        'syntheticsamplesize': [500, 1000, 5000],
        'batchsize': [50, 100],
        'betasend': [0.5, 1.5],
        'n_hidden_D': [128, 256],
        'num_hidden_layers_D': [1, 2],
        'n_hidden_G': [128, 256],
        'num_hidden_layers_G': [1, 2],
        'dpower': [2/5, 4/5],
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    os.system("python3 main.py --dataset tanh_synthetic --sanity_check --VItype implicit --bias --epochs 200 --MCs 100 --betasbegin 0.1 --betalogscale "
              "--taskid %s --syntheticsamplesize %s --dpower %s --batchsize %s "
              "--betasend %s "
              "--n_hidden_D %s --num_hidden_layers_D %s --n_hidden_G %s --num_hidden_layers_G %s"
              %(taskid, temp['syntheticsamplesize'], temp['dpower'], temp['batchsize'],
                temp['betasend'],
                temp['n_hidden_D'], temp['num_hidden_layers_D'], temp['n_hidden_G'], temp['num_hidden_layers_G'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])
