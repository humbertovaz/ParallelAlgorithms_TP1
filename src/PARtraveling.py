# -*- coding: utf-8 -*-
import random
import math
import numpy as np
from traveling import traveling
from traveling2 import traveling2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  # Paralelismo
import multiprocessing
import sys
import time
import getopt



#Vars. Globais
iterations = int(sys.argv[1])
procs = int(sys.argv[2])
cidades = int(sys.argv[3])


#Gerar coordenadas das cidades

def random_integers(num):
    x = [i for i in range(num)]
    random.shuffle(x)
    return x

def random_numbers(num, lower, upper):
    return [10 * random.random() for i in range(num)]

#Calcula as coordenadas de uma rota de cidades -> [x1,x2,x3..] [y1,y2,y3..]
def mapcities(x, y, route1):
    cX = []
    cY = []
    for city in route1:
        cX.append(x[city])
        cY.append(y[city])
    return cX, cY

# Executa o Traveling e Traveling2  e conta os tempos de execucao
def worker(n, D, p):
    dist = [0, 0]
    start = 0
    elapsed = [0, 0]
    route = np.zeros((2, n), dtype=int)
    # traveling 1 - com SA
    #print('Traveling 1 proc', p)
    start = time.clock()  # tira o tempo
    dist[0], route[0] = traveling(D, n, iterations)
    elapsed[0] = (time.clock() - start)  # traveling
    # traveling 2 - sem SA
    #print('Traveling 2 proc', p)
    start = time.clock()  # tira o tempo
    dist[1], route[1] = traveling2(D, n, iterations)  # traveling2
    elapsed[1] = (time.clock() - start)
    return {'dist': dist, 'route': route, 'time': elapsed}


def display_graphs(x, y, min_dist, max_dist, min_route, max_route, proc_min, proc_max, it, n, time_min, time_max):
    #Imagem 1
        plt.rcParams["figure.figsize"] = (8, 8)
        plt.suptitle(str(n) + ' Cidades, ' + str(it) +
                     ' Iteracoes, ' + str(procs) + ' Processos', fontsize=20)
    # Caixeiro c\ Simulated Annealing melhor resultado
        citiesX, citiesY = mapcities(x, y, min_route[0])
        plt.subplot(1, 2, 1)
        #plt.plot(citiesX, citiesY,'ro') -> pontos vermelhos
        plt.plot(citiesX, citiesY, 'k', citiesX, citiesY, 'ro')
        plt.title("Melhor Caixeiro Viajante c/ simul. Annealing - processo " +
                  str(proc_min[0]) + "\nTime: " + str(time_min[0]), fontsize=8)
        plt.legend([str(round(min_dist[0], 4))])

    # Caixeiro c\ Simulated Annealing pior resultado
        citiesX, citiesY = mapcities(x, y, max_route[0])
        plt.subplot(1, 2, 2)
        plt.plot(citiesX, citiesY, 'k', citiesX, citiesY, 'ro')
        plt.title("Pior Caixeiro Viajante c/ simul. Annealing - processo " +
                  str(proc_max[0]) + "\nTime: " + str(time_max[0]), fontsize=8)
        plt.legend([str(round(max_dist[0], 4))])
        plt.show()

    #Imagem 2
        plt.rcParams["figure.figsize"] = (8, 8)
        plt.suptitle(str(n) + ' Cidades, ' + str(it) +
                     ' Iteracoes, ' + str(procs) + ' Processos', fontsize=20)
    # Caixeiro s\ Simulated Annealing melhor resultado
        citiesX, citiesY = mapcities(x, y, min_route[1])
        plt.subplot(1, 2, 1)
        plt.plot(citiesX, citiesY, 'k', citiesX, citiesY, 'ro')
        plt.title("Melhor Caixeiro Viajante Monte Carlo - processo " +
                  str(proc_min[1]) + "\nTime: " + str(time_min[1]), fontsize=8)
        plt.legend([str(round(min_dist[1], 4))])

    # Caixeiro s\ Simulated Annealing pior resultado
        citiesX, citiesY = mapcities(x, y, max_route[1])
        plt.subplot(1, 2, 2)
        plt.plot(citiesX, citiesY, 'k', citiesX, citiesY, 'ro')
        plt.title("Pior Caixeiro Viajante Monte Carlo - processo " +
                  str(proc_max[1]) + "\nTime: " + str(time_max[1]), fontsize=8)
        plt.legend([str(round(max_dist[1], 4))])
        plt.show()

#Imprime resultados para um ficheiro do Traveling2
def printToFileAnnealing(procMin, procMax, timeMin, timeMax, mindist, maxdist):
    orig_stdout = sys.stdout  # guardar o antigo file descriptor
    f = open(str(random.randint(0, 500)) +
             'annealing_' + 'its_' + str(iterations) + '_procs_' + str(procs) + '_cidades_' + str(cidades) + '.txt', 'w')
    sys.stdout = f  # definir o filedescriptor 1 com o novo ficheiro
    print("Caixeiro vianjante com Annealing\n")
    print("Tempo min:" + str(timeMin) +
          " (processo nr " + str(procMin) + ") " + "\n")
    print("Tempo max:" + str(timeMax) +
          " (processo nr:" + str(procMax) + ") " + "\n")
    print("Distancia Percorrida Melhor:" + str(mindist) + "\n")
    print("Distancia Percorrida Pior:" + str(maxdist) + "\n")
    sys.stdout = orig_stdout
    f.close()

#Imprime resultados para um ficheiro do Traveling
def printToFileNotAnnealing(procMin, procMax, timeMin, timeMax, mindist, maxdist):
    orig_stdout = sys.stdout  # guardar o antigo file descriptor
    f = open(str(random.randint(0, 500)) +
             'montecarlo_' + 'its_' + str(iterations) + '_procs_' + str(procs) + '_cidades_' + str(cidades) + '.txt', 'w')
    sys.stdout = f  # definir o filedescriptor 1 com o novo ficheiro
    print("Caixeiro vianjante Monte Carlo\n")
    print("Tempo min:" + str(timeMin) +
          " (processo nr:" + str(procMin) + ") " + "\n")
    print("Tempo max:" + str(timeMax) +
          " (processo nr" + str(procMax) + ") " + "\n")
    print("Distancia Percorrida Melhor:" + str(mindist) + "\n")
    print("Distancia Percorrida Pior:" + str(maxdist) + "\n")
    sys.stdout = orig_stdout
    f.close()

# Gera as coordenadas X e Y das N cidades e as distancias entre elas
def generateCoordinatesAndDist(n):
    # Gerar coordenadas das n cidades
    x = random_numbers(n, 1, 10)  # (numCidades,lower,upper)
    y = random_numbers(n, 1, 10)
    # Declarar a matriz das distancias
    D = [[0 for x in range(n)] for y in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            #and computes the distances between them
            D[i][j] = math.sqrt(math.pow(x[i] - x[j], 2) +
                                math.pow(y[i] - y[j], 2))
    return x, y, D

#Funcao que chama as necessarias para a execução do traveling e traveling2
def PARtraveling(n):
    #Gerar Coordenadas e Distancias
    x, y, D = generateCoordinatesAndDist(n)

    min_dist = [sys.maxsize, sys.maxsize]
    max_dist = [0, 0]
    min_route = np.zeros((2, n + 1), dtype=int)
    max_route = np.zeros((2, n + 1), dtype=int)
    proc_min = [0, 0]
    proc_max = [0, 0]
    time_min = [0, 0]
    time_max = [0, 0]

    # Executar os 2 caixeiros viajantes com #p processos em paralelo
    results = Parallel(n_jobs=procs, backend="threading")(
        delayed(worker)(n, D, p) for p in range(procs))

    # Recolher resultados dos X procs
    for p in range(0, procs):  # X procs
        dist = results[p]['dist']
        route = results[p]['route']
        time = results[p]['time']

        # Determinar os processos c\ dist min e dist maxima
        for i in range(0, 2):
                if dist[i] < min_dist[i]:
                    min_dist[i] = dist[i]
                    # fecha rota (volta à 1ª cidade)
                    min_route[i] = np.append(route[i], route[i][0])
                    proc_min[i] = p
                    time_min[i] = time[i]

                if dist[i] > max_dist[i]:
                    max_dist[i] = dist[i]
                    # fecha rota (volta à 1ª cidade)
                    max_route[i] = np.append(route[i], route[i][0])
                    proc_max[i] = p
                    time_max[i] = time[i]
                    #print("time max ",time_max[i])
    #display_graphs(x, y, min_dist, max_dist, min_route, max_route, proc_min,
    #               proc_max, iterations, n, time_min, time_max)  # 100 predefinido como nr de iteracoes

    printToFileAnnealing(proc_min[0],proc_max[0],time_min[0],time_max[0],min_dist[0],max_dist[0]) # annealing
    printToFileNotAnnealing(proc_min[1],proc_max[1],time_min[1],time_max[1],min_dist[1],max_dist[1]) # not annealing

#Imprime resultados para um ficheiro do GreedyAlgorithm
def printToFileGreedy(procMin, procMax, timeMin, timeMax, mindist, maxdist):
    orig_stdout = sys.stdout  # guardar o antigo file descriptor
    f = open(str(random.randint(0, 500)) +
             'greedy_' + 'its_' + str(iterations) + '_procs_' + str(procs) + '_cidades_' + str(cidades) + '.txt', 'w')
    sys.stdout = f  # definir o filedescriptor 1 com o novo ficheiro
    print("Caixeiro vianjante Greedy\n")
    print("Tempo min: " + str(timeMin) +
          " (processo nr " + str(procMin) + ") " + "\n")
    print("Tempo max: " + str(timeMax) +
          " (processo nr " + str(procMax) + ") " + "\n")
    print("Distancia Percorrida Melhor: " + str(mindist) + "\n")
    print("Distancia Percorrida Pior: " + str(maxdist) + "\n")
    sys.stdout = orig_stdout
    f.close()

# Plot do greedyAlgorithm
def display_greedy(n, x, y, min_route, max_route, proc_min, proc_max, min_dist, max_dist):
    #mapear as cidades da rota
    citiesX1, citiesY1 = mapcities(x, y, min_route)
    #Imagem 1
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.suptitle(str(n) + ' Cidades, ' + str(iterations) + ' Iteracoes, ' + str(procs) + ' Processos', fontsize=20)
    plt.subplot(1, 2, 1)
    plt.plot(citiesX1, citiesY1, 'k', citiesX1, citiesY1, 'ro')
    plt.title("Melhor Caixeiro Viajante Greedy - processo " +
              str(proc_min), fontsize=8)
    plt.legend([str(round(min_dist, 4))])
    
    #mapear as cidades da rota
    citiesX2, citiesY2 = mapcities(x, y, max_route)
    #Imagem 2
    plt.suptitle(str(n) + ' Cidades, ' + str(iterations) + ' Iteracoes, ' + str(procs) + ' Processos', fontsize=20)
    plt.subplot(1, 2, 2)
    plt.plot(citiesX2, citiesY2, 'k', citiesX2, citiesY2, 'ro')
    plt.title("Pior Caixeiro Viajante Greedy - processo " +
                  str(proc_min), fontsize=8)
    plt.legend([str(round(max_dist, 4))])
    plt.show()

# Gera coordenadas, Mat Distancia, e rota das cidades. De seguida cria #procs e corre-os em paralelo
def greedyHandler(n):
    x, y, D = generateCoordinatesAndDist(n)
    
    # gerar caminho aleatório de cidades
    cities = random_integers(n)
    results = Parallel(n_jobs=procs, backend="threading")(
        delayed(workerGreedy)(cities, D, p) for p in range(procs))

    min_dist = sys.maxsize
    max_dist = 0
    min_route = np.zeros((1, n + 1), dtype=int)
    max_route = np.zeros((1, n + 1), dtype=int)
    proc_min = 0
    proc_max = 0
    time_min = sys.maxsize
    time_max = 0
    # Recolher resultados dos X procs
    for p in range(0, procs):  # X procs
        dist = results[p]['dist']
        route = results[p]['route']
        time =  results[p]['time']
        if dist < min_dist:
            min_dist = dist
            min_route = np.append(route, route[0]) # fecha a rota no ponto inicial
            proc_min = p
            time_min = time
        if dist > max_dist:
            max_dist = dist
            max_route = np.append(route, route[0]) # fecha a rota no ponto inicial
            proc_max = p
            time_max = time
    #display_greedy(n,x, y, min_route, max_route, proc_min,
    #               proc_max, min_dist, max_dist)
    printToFileGreedy(proc_min, proc_max, time_min, time_max, min_dist, max_dist)

# Corre o greedy_algorithm de um processo individual
def workerGreedy(cities, D, p):
    # Da como res. o itnerario e a dist. percorrida
    start = time.clock()    
    route, dist = greedy_algorithm(cities, D)
    elapsed = (time.clock() - start)  # traveling

    return {'dist': dist, 'route': route, 'time': elapsed}


def greedy_algorithm(cities, D):
    # Greedy Algorithm
    unvisitedCities = cities [:]
    solution = []
    n = unvisitedCities[0]  # escolhe a 1ª cidade
    unvisitedCities.remove(n)
    solution.append(n)
    totalDist = 0
    while len(unvisitedCities) > 0:
        min_l = None
        min_n = None
        for c in unvisitedCities:
            l = D[n][c]  # Vai buscar a distancia entre as cidades c e n
            if min_l is None:
                min_l = l
                min_n = c
                #totalDist = totalDist + min_l
                #print("min_l: " + str(min_l))
            elif l < min_l:
                min_l = l
                min_n = c
                #totalDist += min_l
                #print("min_l: " + str(min_l))
        solution.append(min_n)
        totalDist = totalDist + min_l
        unvisitedCities.remove(min_n)
        n = min_n  # seleciona outra cidade
    return solution, totalDist



PARtraveling(cidades)   
greedyHandler(cidades)  

#town = np.random.permutation(48)
#for t in town:
#print("town: "+ str(town))
#cities = random_integers(48)
#print("cities: " + str(cities))