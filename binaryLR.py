import numpy as np
import sys
import os


def main(taskid):
    dataset_num = int(taskid[0])
    #dataset_num, mc = np.unravel_index(taskid, [3, 100])

    if dataset_num == 0:
        os.system("python3 RLCT.py --dataset breastcancer-binary --network logistic --epochs 100 --batchsize 10 --betalogscale --bl 100 --MCs 100")
    elif dataset_num == 1:
        os.system("python3 RLCT.py --dataset iris-binary --network logistic --epochs 100 --batchsize 10 --betalogscale --bl 100 --MCs 100")
    else:
        os.system("python3 RLCT.py --dataset MNIST-binary --network logistic --betalogscale --bl 100 --MCs 100")

if __name__ == "__main__":
    main(sys.argv[1:])
