#!/bin/bash
#for N in 20 50 100 200
for N in 200
	do
      for seed in 1 2
            do
            #for net_spec in "exact" "SK_net_rs" "SK_net_rs_set"  "SL" "MADE_21" "MADE_22"
            #for net_spec in "SK_net_rs_set"  "SL" "MADE_21" "MADE_22"
            for net_spec in "SK_0rsb" "SK_1rsb" "SK_2rsb"
                  do
                  echo $N
                  echo $net_spec
                  echo $seed
                  sed "s/N=.*/N=$N/" run_sk.sh > temp0.sh
                  sleep 0.02
                  sed "s/net_spec=.*/net_spec=$net_spec/"  temp0.sh > temp1.sh
                  sleep 0.02
                  sed "s/seed=.*/seed=$seed/"  temp1.sh > temp.sh
                  sleep 0.02
                  ./temp.sh
                  #sbatch temp.sh
                  sleep 0.02
            done
	done
done
