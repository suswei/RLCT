#!/bin/bash
#PBS -N rlct
#PBS -M susan.wei@unimelb.edu.au
#PBS -m abe
#PBS -P uj89
#PBS -q express
#PBS -l walltime=06:00:00
#PBS -l mem=128GB
#PBS -l jobfs=1GB
#PBS -l ncpus=48
#PBS -l wd

module load python3
module load pytorch

for i in {40..41}; do
 python3 sweep.py $i &
done

wait
