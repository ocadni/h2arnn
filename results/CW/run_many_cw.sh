#!/bin/bash
for N in 20 50 100 200
	do
      for net_spec in "exact" "exact_Ninf" "CWARNN" "SL" "1Par" "MADE_21" "MADE_22" "CWARNN_inf"
		do
            echo $N
            echo $net_spec
            sed "s/N=.*/N=$N/" run_cw.sh > temp0.sh
            sleep 0.2
            sed "s/net_spec=.*/net_spec=$net_spec/"  temp0.sh > temp.sh
            sleep 0.2
            chmod +x temp.sh
            #sbatch temp.sh
            ./temp.sh
            sleep 0.2
	done
done