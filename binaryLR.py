import sys
import os

def main(taskid):
    # dataset_num = int(taskid[0])
    #dataset_num, mc = np.unravel_index(taskid, [3, 100])

    taskid = int(taskid[0])

    if taskid == 0:
        os.system("python3 RLCT_IVI.py "
                  "--dataset lr_synthetic "
                  "--syntheticsamplesize 500 "
                  "--network logistic "
                  "--epochs 20 "
                  "--batchsize 50 "
                  "--lambda_asymptotic thm4 "
                  "--betalogscale true "
                  "--wandb_on")
    elif taskid == 1:
        os.system("python3 RLCT_IVI.py "
                  "--dataset lr_synthetic "
                  "--syntheticsamplesize 5000 "
                  "--network logistic "
                  "--epochs 20 "
                  "--batchsize 50 "
                  "--lambda_asymptotic thm4 "
                  "--betalogscale true "
                  "--wandb_on")
    else:
        os.system("python3 RLCT_IVI.py "
                  "--dataset lr_synthetic "
                  "--syntheticsamplesize 50000 "
                  "--network logistic "
                  "--epochs 20 "
                  "--batchsize 50 "
                  "--lambda_asymptotic thm4 "
                  "--betalogscale true "
                  "--wandb_on")


if __name__ == "__main__":
    main(sys.argv[1:])
