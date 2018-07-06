import math
import random
import sys
import time
from joblib import Parallel, delayed  # Paralelismo
import numpy as np
# GA definitivo
iterations = int(sys.argv[1])
procs = int(sys.argv[2])
cidades = int(sys.argv[3])


def random_numbers(num, lower, upper):
    return [10 * random.random() for i in range(num)]


def mapcities(x, y, route1):
    cX = []
    cY = []
    for city in route1:
        cX.append(x[city])
        cY.append(y[city])
    return cX, cY


class Tour:
   def __init__(self, tourmanager, tour=None):
      self.tourmanager = tourmanager
      self.tour = []
      self.fitness = 0.0
      self.distance = 0
      if tour is not None:
         self.tour = tour
      else:
         for i in range(0, self.tourmanager.numberOfCities()):
            self.tour.append(None)

   def __len__(self):
      return len(self.tour)

   def __getitem__(self, index):
      return self.tour[index]

   def __setitem__(self, key, value):
      self.tour[key] = value

   def __repr__(self):
      geneString = "|"
      for i in range(0, self.tourSize()):
         geneString += str(self.getCity(i)) + "|"
      return geneString

   def generateIndividual(self):
      for cityIndex in range(0, self.tourmanager.numberOfCities()):
         self.setCity(cityIndex, self.tourmanager.getCity(cityIndex))
      random.shuffle(self.tour)

   def getCity(self, tourPosition):
      return self.tour[tourPosition]

   def setCity(self, tourPosition, city):
      self.tour[tourPosition] = city
      self.fitness = 0.0
      self.distance = 0

   def getFitness(self):
      if self.fitness == 0:
         self.fitness = 1 / float(self.getDistance())
      return self.fitness

   def getDistance(self):
      if self.distance == 0:
         tourDistance = 0
         for cityIndex in range(0, self.tourSize()):
            fromCity = self.getCity(cityIndex)
            destinationCity = None
            if cityIndex + 1 < self.tourSize():
               destinationCity = self.getCity(cityIndex + 1)
            else:
               destinationCity = self.getCity(0)
            tourDistance += fromCity.distanceTo(destinationCity)
         self.distance = tourDistance
      return self.distance

   def tourSize(self):
      return len(self.tour)

   def containsCity(self, city):
      return city in self.tour


class Population:
    def __init__(self, tourmanager, populationSize, initialise):
        self.tours = []
        for i in range(0, populationSize):
            self.tours.append(None)

        if initialise:
            for i in range(0, populationSize):
                newTour = Tour(tourmanager)
                newTour.generateIndividual()
                self.saveTour(i, newTour)

    def __setitem__(self, key, value):
        self.tours[key] = value

    def __getitem__(self, index):
        return self.tours[index]

    def saveTour(self, index, tour):
        self.tours[index] = tour

    def getTour(self, index):
        return self.tours[index]

    def getFittest(self):
        fittest = self.tours[0]
        for i in range(0, self.populationSize()):
            if fittest.getFitness() <= self.getTour(i).getFitness():
                fittest = self.getTour(i)
        return fittest

    def populationSize(self):
        return len(self.tours)


class City:
        def __init__(self, x=None, y=None):
            self.x = None
            self.y = None
            if x is not None:
                self.x = x
            else:
                self.x = int(random.random() * 200)
            if y is not None:
                self.y = y
            else:
                self.y = int(random.random() * 200)

        def getX(self):
            return self.x

        def getY(self):
            return self.y

        def distanceTo(self, city):
            xDistance = abs(self.getX() - city.getX())
            yDistance = abs(self.getY() - city.getY())
            distance = math.sqrt((xDistance * xDistance) +
                                 (yDistance * yDistance))
            return distance

        def __repr__(self):
            return str(self.getX()) + ", " + str(self.getY())


class GA:
   def __init__(self, tourmanager):
      self.tourmanager = tourmanager
      self.mutationRate = 0.015
      self.tournamentSize = 5
      self.elitism = True

   def evolvePopulation(self, pop):
      newPopulation = Population(self.tourmanager, pop.populationSize(), False)
      elitismOffset = 0
      if self.elitism:
         newPopulation.saveTour(0, pop.getFittest())
         elitismOffset = 1

      for i in range(elitismOffset, newPopulation.populationSize()):
         parent1 = self.tournamentSelection(pop)
         parent2 = self.tournamentSelection(pop)
         child = self.crossover(parent1, parent2)
         newPopulation.saveTour(i, child)

      for i in range(elitismOffset, newPopulation.populationSize()):
         self.mutate(newPopulation.getTour(i))

      return newPopulation

   def crossover(self, parent1, parent2):
      child = Tour(self.tourmanager)

      startPos = int(random.random() * parent1.tourSize())
      endPos = int(random.random() * parent1.tourSize())

      for i in range(0, child.tourSize()):
         if startPos < endPos and i > startPos and i < endPos:
            child.setCity(i, parent1.getCity(i))
         elif startPos > endPos:
            if not (i < startPos and i > endPos):
               child.setCity(i, parent1.getCity(i))

      for i in range(0, parent2.tourSize()):
         if not child.containsCity(parent2.getCity(i)):
            for ii in range(0, child.tourSize()):
               if child.getCity(ii) == None:
                  child.setCity(ii, parent2.getCity(i))
                  break

      return child

   def mutate(self, tour):
      for tourPos1 in range(0, tour.tourSize()):
         if random.random() < self.mutationRate:
            tourPos2 = int(tour.tourSize() * random.random())

            city1 = tour.getCity(tourPos1)
            city2 = tour.getCity(tourPos2)

            tour.setCity(tourPos2, city1)
            tour.setCity(tourPos1, city2)

   def tournamentSelection(self, pop):
      tournament = Population(self.tourmanager, self.tournamentSize, False)
      for i in range(0, self.tournamentSize):
         randomId = int(random.random() * pop.populationSize())
         tournament.saveTour(i, pop.getTour(randomId))
      fittest = tournament.getFittest()
      return fittest


class TourManager:
    destinationCities = []

    def addCity(self, city):
        self.destinationCities.append(city)

    def getCity(self, index):
      return self.destinationCities[index]

    def numberOfCities(self):
      return len(self.destinationCities)


##################################################


def printToFileGA(procMin, procMax, timeMin, timeMax, mindist, maxdist):
    orig_stdout = sys.stdout  # guardar o antigo file descriptor
    f = open(str(random.randint(0, 500)) +
             'ga' + 'its_' + str(iterations) + '_procs_' + str(procs) + '_cidades_' + str(cidades) + '.txt', 'w')
    sys.stdout = f  # definir o filedescriptor 1 com o novo ficheiro
    print("Caixeiro vianjante GA\n")
    print("Tempo min: " + str(timeMin) +
          " (processo nr " + str(procMin) + ") " + "\n")
    print("Tempo max: " + str(timeMax) +
          " (processo nr " + str(procMax) + ") " + "\n")
    print("Distancia Percorrida Melhor: " + str(mindist) + "\n")
    print("Distancia Percorrida Pior: " + str(maxdist) + "\n")
    sys.stdout = orig_stdout
    f.close()

def gaAlgorithm(tourmanager,pop):
    # Evolve population for 50 generations
    ga = GA(tourmanager)
    pop = ga.evolvePopulation(pop)
    for i in range(0, iterations):
        pop = ga.evolvePopulation(pop)
        tour = pop.getTour(i)
        distance = tour.getDistance()
    
    return pop,distance
    
def handlerGA():
    # Create Tour Manager
    tourmanager = TourManager()

    #Create the coordinates for the cities
    x = random_numbers(cidades, 1, 10)  # (numCidades,lower,upper)
    y = random_numbers(cidades, 1, 10)  # (numCidades,lower,upper)
     
    #Create Cities and add them to the tourmanager
    for i in range(0, cidades):
        city = City(x[i], y[i])
        tourmanager.addCity(city)
    
    # Initialize population
    pop = Population(tourmanager, cidades, True)
    
    results = Parallel(n_jobs=procs, backend="threading")(
        delayed(workerGA)(tourmanager,pop) for p in range(procs))
    
    min_dist = sys.maxsize
    max_dist = 0
    min_route = np.zeros((1, cidades + 1), dtype=int)
    max_route = np.zeros((1, cidades + 1), dtype=int)
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
    printToFileGA(proc_min, proc_max, time_min, time_max, min_dist, max_dist)
    

def workerGA(tourmanager,pop):
    start = time.clock()    
    route,dist = gaAlgorithm(tourmanager,pop)
    elapsed = (time.clock() - start)  # traveling

    return {'dist': dist, 'route': route, 'time': elapsed}


class Main():
    if __name__ == '__main__':
        handlerGA()