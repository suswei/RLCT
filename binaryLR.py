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
        os.system("python3 RLCT.py --dataset breastcancer-binary --epochs 100 --batchsize 10 --network logistic")
    elif dataset_num == 1:
        os.system("python3 RLCT.py --dataset iris-binary --epochs 100 --batchsize 10 --network logistic")
    else:
        os.system("python3 RLCT.py  --dataset MNIST-binary --epochs 10 --batchsize 64 --network logistic")

if __name__ == "__main__":
    main(sys.argv[1:])
