#!/bin/bash
#SBATCH --job-name=BPEpi_sens
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.log
#SBATCH --mem=4GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=indaco.biazzo@epfl.ch

beta_init=0.1
beta_end=2
beta_step=100
num_threads=4
lr=0.001
max_step=1000
batch_size=2000
std_fe_limit=1e-4
batch_iter=20
stats_step=1
save_dir="results/Curie-Weiss/data/"
for N in 20 100 200
do
    for net_spec in  "sum_exp" "MADE_22" "exact" "one" "sp2" "sp4" "sum_exp_exact" "MADE" "MADE_21"  "SL" 
    do
        python run.py --N $N --beta_range $beta_init $beta_end $beta_step --net_spec $net_spec --num_threads $num_threads --lr $lr --max_step $max_step --batch_size $batch_size --std_fe_limit $std_fe_limit --batch_iter $batch_iter --stats_step $stats_step --save_dir $save_dir
    done
done