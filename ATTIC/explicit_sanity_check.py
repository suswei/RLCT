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
            'dataset': ['logistic_synthetic', 'tanh_synthetic', 'reducedrank_synthetic'],
            'syntheticsamplesize': [1000],
            'posterior_method': ['explicit'],
            'epochs': [500],
            'batchsize': [10, 50],
            'beta_auto': ['beta_auto_liberal', 'beta_auto_conservative'],
            'betalogscale': ['true'],
            'lambda_asymptotic': ['thm4'],
            'dpower': [2/5, 4/5], #1/5, 2/5, 3/5, 4/5
            'MCs': [10],
            'R': [10],
            'lr': [0.05, 0.01, 0.001]
        }
    elif (taskid-72) in range(72):
        index = 2
        hyperparameter_config = {
            'dataset': ['logistic_synthetic', 'tanh_synthetic', 'reducedrank_synthetic'],
            'syntheticsamplesize': [5000],
            'posterior_method': ['explicit'],
            'epochs': [500],
            'batchsize': [100, 150],
            'beta_auto': ['beta_auto_liberal', 'beta_auto_conservative'],
            'betalogscale': ['true'],
            'lambda_asymptotic': ['thm4'],
            'dpower': [2/5, 4/5], #1/5, 2/5, 3/5, 4/5
            'MCs': [10],
            'R': [10],
            'lr': [0.05, 0.01, 0.001]
        }
    elif (taskid - 72 - 72) in range(72):
        index = 3
        hyperparameter_config = {
            'dataset': ['logistic_synthetic', 'tanh_synthetic', 'reducedrank_synthetic'], #--dataset reducedrank_synthetic --syntheticsamplesize 10000 --batchsize 128 --network reducedrank --betasend 0.5 --epochs 50 --MCs 2 --R 5 --num_hidden_layers_G 4 --num_hidden_layers_D 4 --lr_primal 0.01 --lr_dual 0.005
            'syntheticsamplesize': [10000],
            'posterior_method': ['explicit'],
            'epochs': [500],
            'batchsize': [150, 250],
            'beta_auto': ['beta_auto_liberal', 'beta_auto_conservative'],
            'betalogscale': ['true'],
            'lambda_asymptotic': ['thm4'],
            'dpower': [2 / 5, 4 / 5],  # 1/5, 2/5, 3/5, 4/5
            'MCs': [10],
            'R': [10],
            'lr': [0.05, 0.01, 0.001]
        }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if index == 1:
        i = taskid
    elif index == 2:
        i = taskid - 72
    elif index == 3:
        i = taskid - 72 - 72
    key, value = zip(*hyperparameter_experiments[i].items())

    dataset = value[0]
    if dataset == 'logistic_synthetic':
        network = 'logistic'
    elif dataset == 'tanh_synthetic':
        network = 'tanh'
    elif dataset == 'reducedrank_synthetic':
        network = 'reducedrank'

    syntheticsamplesize = value[1]
    posterior_method = value[2]
    epochs = value[3]
    batchsize = value[4]
    beta_auto = value[5]
    betalogscale = value[6]
    lambda_asymptotic = value[7]
    dpower = value[8]
    MCs = value[9]
    R = value[10]
    lr = value[11]

    os.system("python main.py "
              "--taskid %s "
              "--dataset %s "
              "--syntheticsamplesize %s "
              "--dpower %s "
              "--network %s "
              "--epochs %s "
              "--batchsize %s "
              "--lambda_asymptotic %s "             
              "--posterior_method %s "
              "--lr %s "
              "--%s "
              "--betalogscale %s "
              "--MCs %s "
              "--R %s "%(taskid, dataset, syntheticsamplesize, dpower, network, epochs, batchsize, lambda_asymptotic, posterior_method, lr, beta_auto, betalogscale, MCs, R))

if __name__ == "__main__":
    main(sys.argv[1:])
