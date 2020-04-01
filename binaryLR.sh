#!/usr/bin/env bash

# The name of the job:
#SBATCH --job-name="binaryLR"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-5:0:00

# Batch arrays
#SBATCH --array=0-2

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# Use this email address:
#SBATCH --mail-user=susan.wei@unimelb.edu.au

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The job command(s):
module load web_proxy
wandb login e0b3d65dc819aef2a9857e591d21a200bb350011
python binaryLR.py ${SLURM_ARRAY_TASK_ID}
