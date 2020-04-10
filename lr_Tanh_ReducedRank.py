import sys
import os
import itertools

def main(taskid):
    # dataset_num = int(taskid[0])
    #dataset_num, mc = np.unravel_index(taskid, [3, 100])

    taskid = int(taskid[0])
    if taskid in range(72):
        index = 1
        hyperparameter_config = {
            'dataset': ['lr_synthetic', '3layertanh_synthetic', 'reducedrank_synthetic'],
            'syntheticsamplesize': [500],
            'VItype': ['explicit','implicit'],
            'epochs': [50, 100, 200],
            'batchsize': [10],
            'n_hidden_G': [256],
            'num_hidden_layers_G': [2],
            'n_hidden_D': [256],
            'num_hidden_layers_D': [2],
            'lambda_asymptotic': ['thm4'],
            'betasbegin': [0.1],
            'betasend': [2],
            'betalogscale': ['true'],
            'dpower': [1/5, 2/5, 3/5, 4/5]
        }
    elif (taskid-72) in range(72):
        index = 2
        hyperparameter_config = {
            'dataset': ['lr_synthetic', '3layertanh_synthetic', 'reducedrank_synthetic'],
            'syntheticsamplesize': [5000],
            'VItype': ['explicit','implicit'],
            'epochs': [50, 100, 200],
            'batchsize': [100],
            'n_hidden_G': [256],
            'num_hidden_layers_G': [2],
            'n_hidden_D': [256],
            'num_hidden_layers_D': [2],
            'lambda_asymptotic': ['thm4'],
            'betasbegin': [0.1],
            'betasend': [2],
            'betalogscale': ['true'],
            'dpower': [1/5, 2/5, 3/5, 4/5]
        }
    elif (taskid - 72 - 72) in range(108):
        index = 3
        hyperparameter_config = {
            'dataset': ['lr_synthetic', '3layertanh_synthetic', 'reducedrank_synthetic'],
            'syntheticsamplesize': [10000],
            'VItype': ['explicit', 'implicit'],
            'epochs': [50, 100, 200],
            'batchsize': [200],
            'n_hidden_G': [512],
            'num_hidden_layers_G': [4],
            'n_hidden_D': [512],
            'num_hidden_layers_D': [4],
            'lambda_asymptotic': ['thm4'],
            'betasbegin': [0.1],
            'betasend': [0.5, 1],
            'betalogscale': ['true'],
            'dpower': [1/5, 2/5, 3/5]
        }
    elif (taskid - 72 - 72 - 108) in range(108):
        index = 4
        hyperparameter_config = {
            'dataset': ['lr_synthetic', '3layertanh_synthetic', 'reducedrank_synthetic'],
            'syntheticsamplesize': [60000],
            'VItype': ['explicit', 'implicit'],
            'epochs': [50, 100, 200],
            'batchsize': [256],
            'n_hidden_G': [512],
            'num_hidden_layers_G': [6],
            'n_hidden_D': [512],
            'num_hidden_layers_D': [6],
            'lambda_asymptotic': ['thm4'],
            'betasbegin': [0.1],
            'betasend': [0.2, 0.5],
            'betalogscale': ['true'],
            'dpower': [1/5, 2/5, 3/5]
        }

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if index == 1:
        i = taskid
    elif index == 2:
        i = taskid - 72
    elif index == 3:
        i = taskid - 72 - 72
    elif index == 4:
        i = taskid - 72 - 72 - 108
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
    n_hidden_G = value[5]
    num_hidden_layers_G = value[6]
    n_hidden_D = value[7]
    num_hidden_layers_D = value[8]
    lambda_asymptotic = value[9]
    betasbegin = value[10]
    betasend = value[11]
    betalogscale = value[12]
    dpower = value[13]

    os.system(("python3 main.py "
              "--dataset %s "
              "--syntheticsamplesize %s "
              "--VItype %s "
              "--network %s "
              "--epochs %s "
              "--batchsize %s "
              "--n_hidden_G %s "
              "--num_hidden_layers_G %s "
              "--n_hidden_D %s "
              "--num_hidden_layers_D %s "
              "--lambda_asymptotic %s "
              "--betasbegin %s "
              "--betasend %s "
              "--betalogscale %s "
              "--dpower %s ")%(dataset, syntheticsamplesize, VItype, network, epochs, batchsize, n_hidden_G, num_hidden_layers_G, n_hidden_D, num_hidden_layers_D, lambda_asymptotic, betasbegin, betasend, betalogscale, dpower))

if __name__ == "__main__":
    main(sys.argv[1:])
