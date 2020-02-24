from settings import Settings
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from numpy import copy, dtype


np.set_printoptions(precision=2)  
def read_matrix():
    with  open("city/matrix.txt") as matrixFile:
        resultList = []
        for line in matrixFile:
            line = line.rstrip('\n')
            sVals = line.split(" ")
            iVals = list(map(np.float, sVals))
            resultList.append(iVals)
        matrixFile.close()
    return np.asmatrix(resultList)

matrix = read_matrix()
print(matrix.shape[1] - 1)
maxDemand = matrix.sum()
settings = Settings()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_pos", random.randint, 0, max(matrix.shape)-1)

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
        try:
            if matrixBool.item(individual[t], individual[t+1]) == 0 :
                for i1 in range(-1,2):
                    for i2 in range(-1,2):
                            try:
                                if (individual[t] + i1) < 0 or (individual[t+1] + i2) < 0:
                                    raise IndexError
                                if matrixBool.item(individual[t] + i1, individual[t+1] + i2) == 0 :
                                    surface += matrix.item(individual[t] + i1, individual[t+1] + i2)
                                    matrixBool[individual[t] + i1, individual[t+1] + i2] = 1
                            except IndexError:
                                continue 
        except IndexError:
            return 0, 
    return surface,


def evalSurfaceCoveredMatrixVarDisperse(individual):
    #print(individual)
    matrixCopy = copy(matrix)
    surface = 0
    volume = 0
    
    for t in range(0, settings.individualLength - 1, 2):
        for i1 in range(-2,3):
            for i2 in range(-2,3):
                    try:
                        if (individual[t] + i1) < 0 or (individual[t+1] + i2) < 0:
                            raise IndexError
                        TileCopy = matrixCopy.item(individual[t] + i1, individual[t+1] + i2)
                        TileOriginal = matrix.item(individual[t] + i1, individual[t+1] + i2)
                        
                        if i1 == 0 and i2 == 0:
                            volume = (settings.volumeCenter * TileOriginal) / 100
                        else:
                            if i1 > 1 or i1 < -1 or i2 > 1 or i2 < -1:
                                volume = (settings.volumeTwoTilesAway * TileOriginal) / 100
                            else:
                                volume = (settings.volumeOneTileAway * TileOriginal) / 100
                        
                        if TileCopy > 0:       
                            if volume > TileCopy:
                                surface +=  TileCopy
                            else:
                                surface += volume
                        
                        matrixCopy[individual[t] + i1, individual[t+1] + i2] -= volume
                        #print(str(matrixCopy[individual[t] + i1, individual[t+1] + i2]) + "  ")
                    except IndexError:
                        continue 
    
    return surface * 100 / maxDemand,

def showBestIndividual(individual):
    matrixCopy = copy(matrix)
    matrixVolume = np.full(matrix.shape, 0, dtype=float)
    volume = 0
    
    for t in range(0, settings.individualLength - 1, 2):
        individualVolume = 0
        for i1 in range(-2,3):
            for i2 in range(-2,3):
                try:
                    if (individual[t] + i1) < 0 or (individual[t+1] + i2) < 0:
                        raise IndexError
                    TileCopy = matrixCopy.item(individual[t] + i1, individual[t+1] + i2)
                    TileOriginal = matrix.item(individual[t] + i1, individual[t+1] + i2)
                        
                    if i1 == 0 and i2 == 0:
                        volume = (settings.volumeCenter * TileOriginal) / 100
                    else:
                        if i1 > 1 or i1 < -1 or i2 > 1 or i2 < -1:
                            volume = (settings.volumeTwoTilesAway * TileOriginal) / 100
                        else:
                            volume = (settings.volumeOneTileAway * TileOriginal) / 100
                        
                    if TileCopy > 0:       
                        if volume > TileCopy:
                            individualVolume +=  TileCopy
                        else:
                            individualVolume += volume
                    matrixCopy[individual[t] + i1, individual[t+1] + i2] -= volume
                except IndexError:
                    continue 
        matrixVolume[individual[t], individual[t+1]]= individualVolume
    
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            print("%.2f " % matrixCopy[i][j], end='')
        print("            ", end='')
        for j in range(0, matrix.shape[1]):
            print("%.2f " % matrixVolume[i][j], end='')
        print()
       
def checkBounds():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(0, len(child)-1, 2):
                    if child[i] > matrix.shape[0] - 1:
                        child[i] = random.randint(0, matrix.shape[0] - 1)
                    if child[i+1] > matrix.shape[1] - 1:
                        child[i+1] = random.randint(0, matrix.shape[1] - 1) 
            return offspring
        return wrapper
    return decorator

toolbox.register("evaluate", evalSurfaceCoveredMatrixVarDisperse)

toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", tools.mutUniformInt, low=0, up=max(matrix.shape), indpb=0.05)

toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.decorate("mate", checkBounds())

toolbox.decorate("mutate", checkBounds())

toolbox.decorate("population", checkBounds())
        
def main():
    random.seed()
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
    
    while max(fits) < 100 and g < 50:
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
        
        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s with score %.4f%c" % (best_ind, best_ind.fitness.values[0],'%'))
        showBestIndividual(best_ind)
        print("-- End of (successful) evolution --")
    
    
main()
    