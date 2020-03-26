import numpy as np
import sys
import os

# the followign parameters are computationally intensive but should be set as high as possible
#   --R
#   --epochs
#   --bl
#   --MCs

# unclear factors at this point
#   --betalogscale
#   --betasbegin
#   --betasend
#   --fit_lambda_over_average
#   --prior

def main(taskid):
    dataset_num = int(taskid[0])
    #dataset_num, mc = np.unravel_index(taskid, [3, 100])

    if dataset_num == 0:
        os.system("python3 RLCT_IVI.py --dataset breastcancer-binary --network logistic --epochs 100 --batchsize 10 --bl 20 --R 100 --MCs 10 --lambda_asymptotic cor3 --epsilon_mc 100 --wandb_on")
    elif dataset_num == 1:
        os.system("python3 RLCT_IVI.py --dataset iris-binary --network logistic --epochs 100 --batchsize 10 --bl 20 --R 100 --MCs 10 --lambda_asymptotic cor3 --epsilon_mc 100 --wandb_on")
    else:
        os.system("python3 RLCT_IVI.py  --dataset MNIST-binary --network logistic --epochs 100 --batchsize 10 --bl 20 --R 100 --MCs 10 --lambda_asymptotic cor3 --epsilon_mc 100 --wandb_on")
if __name__ == "__main__":
    main(sys.argv[1:])
