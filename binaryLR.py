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
#   --prior

def main(taskid):
    dataset_num = int(taskid[0])
    #dataset_num, mc = np.unravel_index(taskid, [3, 100])

    if dataset_num == 0:
        os.system("python3 RLCT.py --dataset breastcancer-binary --network logistic --batchsize 10 --betalogscale --epochs 100 --R 100 --bl 100 --MCs 100")
    elif dataset_num == 1:
        os.system("python3 RLCT.py --dataset iris-binary --network logistic --batchsize 10 --betalogscale --epochs 100 --R 100 --bl 100 --MCs 100")
    else:
        os.system("python3 RLCT.py --dataset MNIST-binary --network logistic --betalogscale --epochs 10 --R 100 --bl 100 --MCs 100")

if __name__ == "__main__":
    main(sys.argv[1:])
