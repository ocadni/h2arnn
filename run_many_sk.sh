#!/bin/bash
#for N in 20 50 100 200
for N in 200
	do
	#for net_spec in "exact" "SK_net_rs" "SK_net_rs_set"  "SL" "MADE_21" "MADE_22"
	#for net_spec in "SK_net_rs_set"  "SL" "MADE_21" "MADE_22"
	for net_spec in "_SK_net_1rs_fixed"  "_SK_net_0rs_fixed"
		do
            echo $N
            echo $net_spec
            sed "s/N=.*/N=$N/" run_sk.sh > temp0.sh
            sleep 0.02
            sed "s/net_spec=.*/net_spec=$net_spec/"  temp0.sh > temp.sh
            sleep 0.02
            ./temp.sh
            #sbatch temp.sh
            sleep 0.02
	done
done
