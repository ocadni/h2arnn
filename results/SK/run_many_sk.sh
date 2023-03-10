#!/bin/bash
for N in 20 50 100 200
	do
      for seed in 1
            do
            for net_spec in "SK_0rsb" "SK_1rsb" "SK_2rsb" "SL" "MADE_23" "MADE_32"
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
                  chmod +x temp.sh
                  ./temp.sh
                  #sbatch temp.sh
                  sleep 0.02
            done
	done
done
