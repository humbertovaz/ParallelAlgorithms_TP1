#!/bin/bash
source .bashrc

#Iteracoes: 10, 100, 1000, 100 000
#Processos: 2, 4, 8, 16, 32, 48
it=100
Procs=(2 4 8 16 32 48)
Cidades=(48 160 600)

    
 for proc in ${Procs[*]}
 do
            for cid in ${Cidades[*]}
            do
                echo  "Iteracao $it Processo $proc cidade $cid"
                python PARtraveling.py $it $proc $cid
               # python gadef.py $it $proc $cid
            done
 done
        

mkdir greedy781


mv *greedy_* greedy781



