# --dataset-name breastcancer-binary --network logistic --batch-size 10 --beta1 0.25 --beta2 0.75
#
# --dataset-name iris-binary --network logistic --batch-size 10 --beta1 1 --beta2 1.05
#
# --dataset-name MNIST-binary --network logistic --beta1 1 --beta2 1.05

import numpy as np
import sys
import os


def main(taskid):
    taskid = int(taskid[0])
    dataset_num, mc = np.unravel_index(taskid, [3, 100])

    if dataset_num == 0:
        os.system("python3 RLCT.py --dataset-name breastcancer-binary --network logistic --batch-size 10 --beta1 0.25 --beta2 0.75")
    elif dataset_num == 1:
        os.system("python3 RLCT.py --dataset-name iris-binary --network logistic --batch-size 10 --beta1 1 --beta2 1.5")
    else:
        os.system("python3 RLCT.py --dataset-name MNIST-binary --network logistic --epochs 10 --beta1 0.25 --beta2 0.4")

if __name__ == "__main__":
    main(sys.argv[1:])
