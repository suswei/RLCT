import sys
import os
import itertools

# def main(taskid):
#
#     taskid = int(taskid[0])
#     hyperparameter_config = {
#         'dataset': ['reducedrank_synthetic', 'tanh_synthetic'],
#         'syntheticsamplesize': [10000, 50000],
#         'MCs': 50*[1]
#     }
#     keys, values = zip(*hyperparameter_config.items())
#     hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
#     temp = hyperparameter_experiments[taskid]
#
#     os.system("python3 main.py --sanity_check --dpower 0.6 "
#               "--VItype implicit --epochs 200 --MCs 1 "
#               "--betasbegin 0.1 --betasend 0.5 --betanscale --numbetas 10 "
#               "--taskid %s --dataset %s --syntheticsamplesize %s "
#               "--n_hidden_G 256 --num_hidden_layers_G 2 "
#               "--n_hidden_D 8 --pretrainDepochs 2 --trainDepochs 2"
#               %(taskid, temp['dataset'], temp['syntheticsamplesize']))

def main(taskid):

    taskid = int(taskid[0])
    hyperparameter_config = {
        'symmetry_factor': [3,4,5,6,7,8,9,10],
        'beta': [1, 1.1]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    os.system("python3 pyro_example.py --symmetry-factor %s --beta %s"
              %(temp['symmetry_factor'], temp['beta']))


if __name__ == "__main__":
    main(sys.argv[1:])
