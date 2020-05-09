#!/bin/bash
#PBS -P uj89
#PBS -N rlct
#PBS -M susan.wei@unimelb.edu.au
#PBS -m abe
#PBS -q normal
#PBS -l ncpus=288
#PBS -l mem=10GB
#PBS -l jobfs=1GB
#PBS -l walltime=48:00:00
#PBS -l wd

module load python3
module load pytorch

for i in {0..4799}; do
 python3 sweep.py $i > $PBS_JOBID_$i.log &
done

wait
