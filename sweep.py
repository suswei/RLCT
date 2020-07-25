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
        'symmetry_factor': list(range(3, 11)),# 3 to 10
        'num_hidden': [10, 15, 20],
        'num_data': [100,250,500],
        'mc': list(range(10))
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    temp = hyperparameter_experiments[taskid]

    os.system("python3 pyro_example.py --symmetry-factor %s --num-hidden %s --num-data %s --mc %s"
              %(temp['symmetry_factor'],temp['num_hidden'],temp['num_data'],temp['mc']))


if __name__ == "__main__":
    main(sys.argv[1:])
