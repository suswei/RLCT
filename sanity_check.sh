#!/bin/bash
for num in {0..215} ; do
echo sbatch -p physical --job-name lihui10098777.${num} --account punim0890 --ntasks=1 --cpus-per-task=2 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --begin=now+$((10*num)) --time=2-5:0:00 -o \"slurm_output/slurm-%A_$num.out\" --wrap=\"module load miniconda\; source activate RLCT\; module load web_proxy\; python sanity_check.py $num\"
sleep 1

sbatch -p physical --job-name lihui10098777.${num} --account punim0890 --ntasks=1 --cpus-per-task=2 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --begin=now+$((10*num)) --time=2-5:0:00 -o "slurm_output/slurm-%A_$num.out" --wrap="module load miniconda; source activate RLCT; module load web_proxy; python sanity_check.py $num"
sleep 1
done
echo "All jobs submitted!\n"