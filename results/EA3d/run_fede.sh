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
beta_end=4
beta_step=79
num_threads=2
lr=0.001
max_step=100
batch_size=2000
std_fe_limit=1e-4
batch_iter=20
stats_step=1
init_step=1000
save_dir="./results/EA3d/fede_data/"
N=125
net_spec="SL"
#net_spec="MADE_23"
model="from_file"
device="cpu"
save_net="yes"
seed=0
sparse="true"
file_couplings="results/EA3d/fede_data/16/exv1_NX16_NY16_seed955461039_LAR_JGT_output.txt"
cd ../../
$python -m $script --model $model --N $N --beta_range $beta_init $beta_end $beta_step --net_spec $net_spec --num_threads $num_threads --lr $lr --max_step $max_step --batch_size $batch_size --std_fe_limit $std_fe_limit --batch_iter $batch_iter --stats_step $stats_step --save_dir $save_dir --device $device --save_net $save_net --seed $seed --file_couplings $file_couplings --init_step $init_step --sparse $sparse
cd results/EA3d/