import random
import numpy as np
import math

def traveling(D,n, iterations):
    town = np.random.permutation(n)
    Tdist = D[town[n-1]][town[0]]
    for i in range(0,n-2):
        Tdist = Tdist + D[town[i]][town[i+1]]
    T = 1 # initial "temperature"
    for i in range(0,iterations):
        c = random.randint(0, n - 1)
        if c == 0:
            previous = n -1
            next1 = 1
            next2 = 2

        elif c == n - 2:
            previous = n - 3
            next1 = n - 1
            next2 = 0

        elif c == n - 1:
            previous = n - 2  
            next1 = 0
            next2 = 1

        else: 
            previous = c - 1
            next1 = c + 1
            next2 = c + 2
            
        # delta=increment in length of route
        delta = D[town[previous]][town[next1]] + D[town[c]][town[next2]] - D[town[previous]][town[c]] - D[town[next1]][town[next2]]
        # accept or discard change to route
        if delta < 0 or math.exp(-delta/T) >= random.uniform(0,1):
            # swap
            temp = town[c]
            town[c] = town[next1]
            town[next1] = temp
            Tdist = Tdist + delta
            if delta != 0:
                i = 0
        else: i = i + 1

        T=0.999 * T
    return Tdist,town
