import sys
import os
import itertools


def main(taskid):
    # dataset_num = int(taskid[0])
    #dataset_num, mc = np.unravel_index(taskid, [3, 100])

    taskid = int(taskid[0])
    if taskid in range(54):
        index = 1
        hyperparameter_config = {
            'dataset': ['lr_synthetic', '3layertanh_synthetic', 'reducedrank_synthetic'],
            'syntheticsamplesize': [1000],
            'VItype': ['implicit'],
            'epochs': [20, 50, 100],
            'batchsize': [10],
            'betasbegin': [0.1],
            'betasend': [2],
            'betalogscale': ['true'],
            'n_hidden_D': [256],
            'num_hidden_layers_D': [2],
            'n_hidden_G': [256],
            'num_hidden_layers_G': [2],
            'lambda_asymptotic': ['thm4'],
            'dpower': [2/5], #1/5, 2/5, 3/5, 4/5
            'MCs': [2],
            'R': [10],
            'lr_primal': [0.01, 0.005, 0.001],
            'lr_dual': [0.005, 0.001]
        }
    elif (taskid-54) in range(54):
        index = 2
        hyperparameter_config = {
            'dataset': ['lr_synthetic', '3layertanh_synthetic', 'reducedrank_synthetic'],
            'syntheticsamplesize': [5000],
            'VItype': ['implicit'],
            'epochs': [20, 50, 100],
            'batchsize': [100],
            'betasbegin': [0.1],
            'betasend': [2],
            'betalogscale': ['true'],
            'n_hidden_D': [256],
            'num_hidden_layers_D': [2],
            'n_hidden_G': [256],
            'num_hidden_layers_G': [2],
            'lambda_asymptotic': ['thm4'],
            'dpower': [2/5], #1/5, 2/5, 3/5, 4/5
            'MCs': [2],
            'R': [10],
            'lr_primal': [0.01, 0.005, 0.001],
            'lr_dual': [0.005, 0.001]
        }
    elif (taskid - 54 - 54) in range(54):
        index = 3
        hyperparameter_config = {
            'dataset': ['lr_synthetic', '3layertanh_synthetic', 'reducedrank_synthetic'], #--dataset reducedrank_synthetic --syntheticsamplesize 10000 --batchsize 128 --network ReducedRankRegression --betasend 0.5 --epochs 50 --MCs 2 --R 5 --num_hidden_layers_G 4 --num_hidden_layers_D 4 --lr_primal 0.01 --lr_dual 0.005
            'syntheticsamplesize': [10000],
            'VItype': ['implicit'],
            'epochs': [50, 100, 200],
            'batchsize': [128],
            'betasbegin': [0.1],
            'betasend': [0.5],
            'betalogscale': ['true'],
            'n_hidden_D': [256],
            'num_hidden_layers_D': [4],
            'n_hidden_G': [256],
            'num_hidden_layers_G': [4],
            'lambda_asymptotic': ['thm4'],
            'dpower': [2/5],
            'MCs': [2],
            'R': [10],
            'lr_primal': [0.01, 0.005, 0.001],
            'lr_dual': [0.005, 0.001]
        }
    elif (taskid - 54 - 54 - 54) in range(54):
        index = 4
        hyperparameter_config = {
            'dataset': ['lr_synthetic', '3layertanh_synthetic', 'reducedrank_synthetic'],
            'syntheticsamplesize': [60000],
            'VItype': ['implicit'],
            'epochs': [50, 100, 200],
            'batchsize': [256],
            'betasbegin': [0.1],
            'betasend': [0.2, 0.5],
            'betalogscale': ['true'],
            'n_hidden_D': [512],
            'num_hidden_layers_D': [6],
            'n_hidden_G': [512],
            'num_hidden_layers_G': [6],
            'lambda_asymptotic': ['thm4'],
            'dpower': [2/5], #1/5, 2/5, 3/5
            'MCs': [2],
            'R': [10],
            'lr_primal': [0.01, 0.005, 0.001],
            'lr_dual': [0.005, 0.001]
        }

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if index == 1:
        i = taskid
    elif index == 2:
        i = taskid - 54
    elif index == 3:
        i = taskid - 54 - 54
    elif index == 4:
        i = taskid - 54 - 54 - 54
    key, value = zip(*hyperparameter_experiments[i].items())
    dataset = value[0]
    if dataset == 'lr_synthetic':
        network = 'logistic'
    elif dataset == '3layertanh_synthetic':
        network = 'Tanh'
    elif dataset == 'reducedrank_synthetic':
        network = 'ReducedRankRegression'

    syntheticsamplesize = value[1]
    VItype = value[2]
    epochs = value[3]
    batchsize = value[4]
    betasbegin = value[5]
    betasend = value[6]
    betalogscale = value[7]
    n_hidden_D = value[8]
    num_hidden_layers_D = value[9]
    n_hidden_G = value[10]
    num_hidden_layers_G = value[11]
    lambda_asymptotic = value[12]
    dpower = value[13]
    MCs = value[14],
    R = value[15],
    lr_primal = value[16],
    lr_dual = value[17]

    os.system(("python3 main.py "
              "--taskid %s "
              "--dataset %s "
              "--syntheticsamplesize %s "
              "--VItype %s "
              "--network %s "
              "--epochs %s "
              "--batchsize %s "
              "--betasbegin %s "
              "--betasend %s "
              "--betalogscale %s "
              "--n_hidden_D %s "
              "--num_hidden_layers_D %s " 
              "--n_hidden_G %s "
              "--num_hidden_layers_G %s "              
              "--lambda_asymptotic %s "              
              "--dpower %s "
              "--MCs %s "
              "--R %s "
              "--lr_primal %s "
              "--lr_dual %s ")%(taskid, dataset, syntheticsamplesize, VItype, network, epochs, batchsize, betasbegin, betasend, betalogscale, n_hidden_D, num_hidden_layers_D, n_hidden_G, num_hidden_layers_G, lambda_asymptotic, dpower, MCs, R, lr_primal, lr_dual))

if __name__ == "__main__":
    main(sys.argv[1:])
