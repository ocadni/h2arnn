#!/bin/bash
#SBATCH --job-name=run
#SBATCH --time=34:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=run.log
#SBATCH --mem=3GB
#SBATCH --mail-type=ALL

python="python"
script="python_lib.run"
beta_init=0.1
beta_end=2
beta_step=39
num_threads=2
lr=0.001
max_step=1000
batch_size=2000
std_fe_limit=1e-4
batch_iter=20
stats_step=1
save_dir="./results/SK/data/"
N=20
net_spec="SK_0rsb"
model="SK"
device="cpu"
save_net="yes"
seed=0
cd ../../
$python -m $script --model $model --N $N --beta_range $beta_init $beta_end $beta_step --net_spec $net_spec --num_threads $num_threads --lr $lr --max_step $max_step --batch_size $batch_size --std_fe_limit $std_fe_limit --batch_iter $batch_iter --stats_step $stats_step --save_dir $save_dir --device $device --save_net $save_net --seed $seed
cd results/SK/