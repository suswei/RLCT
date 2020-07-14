#!/bin/bash
#PBS -P uj89
#PBS -N pyro
#PBS -M susan.wei@unimelb.edu.au
#PBS -m abe
#PBS -q normal
#PBS -l ncpus=288
#PBS -l mem=384GB
#PBS -l jobfs=1GB
#PBS -l walltime=48:00:00
#PBS -l wd

# ncpus should be in multiples of 48 for normal queue
# max memory is 64GB per node

module load python3
module load pytorch

for i in {0..15}; do
 python3 sweep.py $i > $PBS_JOBID_$i.log &
done

wait
