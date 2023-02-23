#!/bin/bash
for N in 20 50 100 200
	do
	for net_spec in "exact" "one" "sum_exp" "sp2" "sp4" "sum_exp_exact" "MADE" "MADE_21" "MADE_22" "SL" 
		do
            echo $N
            echo $net_spec
            sed "s/N=.*/N=$N/" run.sh > temp0.sh
            sleep 0.02
            sed "s/net_spec=.*/net_spec=$net_spec/"  temp0.sh > temp.sh
            sleep 0.02
            sbatch temp.sh
            sleep 0.02
	done
done
