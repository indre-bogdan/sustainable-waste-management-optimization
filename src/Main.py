from settings import Settings
import copy
import random

import numpy as np

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


def read_matrix():
    with  open("city/matrix.txt") as matrixFile:
        resultList = []
        for line in matrixFile:
            line = line.rstrip('\n')
            sVals = line.split(" ")
            iVals = list(map(np.int, sVals))
            resultList.append(iVals)
        matrixFile.close()
    return np.asmatrix(resultList)
matrix = read_matrix()
settings = Settings()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

#def randomPos(row_length, column_length):
  #  return (random.randint(0, row_length), random.randint(0, column_length))

toolbox.register("attr_pos", random.randint, 0, matrix.shape[0]-1)

toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_pos, settings.individualLength)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalSurfaceCovered(individual):
    
    matrixBool = np.full(matrix.shape, 0)
    surface = 0
    
    for t in range(0, settings.individualLength-1, 2):
        if matrixBool.item(individual[t], individual[t+1]) == 0 :
            surface += matrix.item(individual[t], individual[t+1])
            matrixBool[individual[t], individual[t+1]] = 1
    return surface,

def evalSurfaceCoveredMatrix(individual):
    
    matrixBool = np.full(matrix.shape, 0)
    surface = 0
    
    for t in range(0, settings.individualLength - 1, 2):
        if matrixBool.item(individual[t], individual[t+1]) == 0 :
            for i1 in range(-1,2):
                for i2 in range(-1,2):
                        try:
                            if matrixBool.item(individual[t] + i1, individual[t+1] + i2) == 0 :
                                surface += matrix.item(individual[t] + i1, individual[t+1] + i2)
                                matrixBool[individual[t] + i1, individual[t+1] + i2] = 1
                        except IndexError:
                            continue 
    return surface,

def evalSurfaceCoveredMatrixVar(individual):
    
    matrixBool = np.full(matrix.shape, 0)
    surface = 0
    
    for t in range(0, settings.individualLength - 1, 2):
        if matrixBool.item(individual[t], individual[t+1]) == 0 :
            for i1 in range(-1,2):
                for i2 in range(-1,2):
                        try:
                            if matrixBool.item(individual[t] + i1, individual[t+1] + i2) == 0 :
                                surface += matrix.item(individual[t] + i1, individual[t+1] + i2)
                                matrixBool[individual[t] + i1, individual[t+1] + i2] = 1
                        except IndexError:
                            continue 
    return surface,

toolbox.register("evaluate", evalSurfaceCoveredMatrix)

toolbox.register("mate", tools.cxTwoPoint)

def mutUniformTuple(individual):
    return (tools.mutUniformInt(individual, low=0, up=7, indpb=0.05), tools.mutUniformInt(individual, low=0, up=8, indpb=0.05))

toolbox.register("mutate", mutUniformTuple)

toolbox.register("select", tools.selTournament, tournsize=3)

        
def main():
    random.seed(69)
    pop = toolbox.population(n=300)
    
    CXPB, MUTPB = 0.5, 0.2
    
    print("Start of evolution")
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    while max(fits) < 500 and g < 50:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
main()
    