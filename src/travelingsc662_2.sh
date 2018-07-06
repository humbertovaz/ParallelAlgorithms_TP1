#!/bin/bash
source .bashrc
#Iteracoes: 10, 100, 1000, 100 000
#Processos: 2, 4, 8, 16, 32, 48
it=100
Procs=(2 4 8 16 32 48)
Cidades=3200

    
        for proc in ${Procs[*]}
        do
                echo  "Iteracao $it Processo $proc cidade $cid"
                python PARtraveling.py $it $proc $cid
                python gadef.py $it $proc $cid  
        done
        

mkdir montecarlo781
mkdir annealing781
mkdir greedy781
mkdir ga781

mv *montecarlo_* montecarlo662
mv *greedy_* greedy662
mv *annealing_* annealing662
mv *gaits_* ga662

