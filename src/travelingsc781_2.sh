#!/bin/bash
source .bashrc
mkdir humberto_greedy_2
cd humberto_greedy_2
cp ../PARtraveling.py humberto_greedy_2
cp ../traveling.py humberto_greedy_2
cp ../traveling2.py humberto_greedy_2

#Iteracoes: 10, 100, 1000, 100 000
#Processos: 2, 4, 8, 16, 32, 48
it=100
Procs=(2 4 8 16 32 48)
Cidades=(3200)

    
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

