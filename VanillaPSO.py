# import utils
import pyDOE
import numpy as np
from pyDOE import lhs
import random
# import fitness_function
import pandas as pd
import multiprocessing as mp
from matplotlib import animation
from matplotlib import pyplot as plt
import time as tm
import copy
import singleObjFuncs
import statistics

# problem setup parameters
variable_bounds = np.array([[2.048,2.048],[-2.048,-2.048]]) # max and min range of the dimension
listOfPrecisionDigits = [3,3] # precision of digits along that dimension
num_var = len(listOfPrecisionDigits)
numberOfRuns = 10 # number of runs of each optimisation to average the performance

# data generation for plotting
# nx, ny = (3, 2)
numpoints = 1000
x = np.linspace(variable_bounds[1][0], variable_bounds[0][0])
y = np.linspace(variable_bounds[1][1], variable_bounds[0][1],numpoints)
xv, yv = np.meshgrid(x, y)
dinput = np.concatenate((xv.view(-1,1),yv.view(-1,1)),axis = 1)
plottingData = singleObjFuncs.Rastrigin(populationPositions)

# Optimisation Parameters ##################
num_init_designs = 100 # population size
maxNumIterations = 100 # number of iterations to run the optimisation 
C1 = 1.9 # this parameter moves the particle towards its past hostoric best
C2 = 1.5 # this parameter moves the particle towards the best among all particles histories
inertia_parameter = 0.5 # this partice mainintains current velocity, aids exploration

# storing data for each run
GlobalBestList = []
BestParametersList = []

for run in range(numberOfRuns):
    print("run number",run)
    # storing stuff
    GlobalBestObjFuncValueList = []
    GlobalBestParameterList = []

    # plotting essestals
    # fig = plt.figure()
    fig, (ax,ax1) =  plt.subplots(1,2)
    # ax = plt.gca()
    jet= plt.get_cmap('viridis')    
    colors = iter(jet(np.linspace(0,1,maxNumIterations)))

    ## initialisation
    # initialise the random particles at different points in the variable space.
    # latin hypercube sampling used for efficient sampling
    # returns samples in the range 0 to 1
    init_designs = np.array(lhs(num_var, num_init_designs))

    # scale 
    # the population is scaled based on the min and mix values of that dimension
    varRange = variable_bounds[0]-variable_bounds[1]
    populationPositions = init_designs*varRange
    populationPositions = populationPositions + variable_bounds[1]

    # round to required digits
    for varNo,limit in enumerate(listOfPrecisionDigits):
        populationPositions[:,varNo] = np.around(populationPositions[:,varNo],limit)

    # Perform first function evaluation for all particles
    # here Rastrigin is the objective function to be minimised
    # its a benchmark function. more stored in singleObjFuncs
    # ****codesmell**** need to change the objective function name below too. 
    # Use list of function handles instead
    functionEval = singleObjFuncs.Rastrigin(populationPositions) 
    # constrainViolation is the net amount by which each particle violates given constraints
    ConstraintViolation = singleObjFuncs.constraintViolation(populationPositions)
    # inintilasing velocity vectors to zero
    velocity = np.zeros((len(populationPositions),1))

    # current population is stored as a dictionary
    # positions are the different particles variable values.
    # objectove function values for each particle
    # Constraint vilation for each particle
    # Current velocit of each particle
    CurrentPopulation = {"Positions":populationPositions,"FunctionValues":functionEval,
    "ConstraintViolation":ConstraintViolation,"Velocities":velocity}
    # print("CurrentPopulation",CurrentPopulation)
    BestPopulation = copy.deepcopy(CurrentPopulation)

    # find index of global best particle
    # Definition of best particle: particle with best function value, among those with minimum constraint vilation
    # lexsort sorts data based on two pr more colummns of data
    CurrentGlobalBestIndex = np.lexsort((CurrentPopulation["FunctionValues"],\
        CurrentPopulation["ConstraintViolation"]))[0]
    PastGlobalBest = [CurrentPopulation["Positions"][CurrentGlobalBestIndex],\
        CurrentPopulation["FunctionValues"][CurrentGlobalBestIndex],\
            CurrentPopulation["ConstraintViolation"][CurrentGlobalBestIndex]]
    # PastGlobalBest is the best particle, its function evaluation and constaint vilation
    # right after inintialisation and one function evaluation

    iteration = 0
    while iteration < maxNumIterations:
        print("iteration number",iteration)
        # # plotting stuff
        ax.clear()
        plt.ion()
        # plt.xlim(-10,10)
        # plt.ylim(-10,10)
        ax.scatter(CurrentPopulation["Positions"][:,0],CurrentPopulation["Positions"][:,1],color=next(colors))
        ax.set_xlim([-5,5])
        ax.set_ylim([-5,5])
        # ax1.scatter([np.arange(len(GlobalBestObjFuncValueList))],GlobalBestObjFuncValueList)
        ax1.plot(np.arange(len(GlobalBestObjFuncValueList)),GlobalBestObjFuncValueList)
        # ax1.set_xlim([-10,10])
        # ax1.set_ylim([-10,10])

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()

        # personal best update
        # for all particles:
            # update the current best of each particle based on 
            # current function values and past best value
        ConstraintViolationCondition = BestPopulation["ConstraintViolation"]>=CurrentPopulation["ConstraintViolation"]
        FunctionValueCondition = BestPopulation["FunctionValues"]>=CurrentPopulation["FunctionValues"]
        RequiredCondition = ConstraintViolationCondition & FunctionValueCondition
        RequiredCondition = np.tile(RequiredCondition,(1,num_var))

        # replacing Best population by updated current population 
        BestPopulation["Positions"][RequiredCondition] = CurrentPopulation["Positions"][RequiredCondition]

        #Global Best Update
        particleOrdering = np.lexsort((np.squeeze(CurrentPopulation["FunctionValues"]),\
            np.squeeze(CurrentPopulation["ConstraintViolation"])))
        CurrentGlobalBestIndex = particleOrdering[0]
        if CurrentPopulation["ConstraintViolation"][CurrentGlobalBestIndex] <= PastGlobalBest[2]:
            if CurrentPopulation["FunctionValues"][CurrentGlobalBestIndex] < PastGlobalBest[1]:
                PastGlobalBest = [CurrentPopulation["Positions"][CurrentGlobalBestIndex],\
                    CurrentPopulation["FunctionValues"][CurrentGlobalBestIndex],\
                        CurrentPopulation["ConstraintViolation"][CurrentGlobalBestIndex]]

        GlobalBestObjFuncValueList.append(PastGlobalBest[1][0])
        GlobalBestParameterList.append(PastGlobalBest[0])

        # velocity update
        # compute velocity based on position of personal best and global best position
        CurrentPopulation["Velocities"] = inertia_parameter*CurrentPopulation["Velocities"] + \
        C1*np.random.random()*(BestPopulation["Positions"]-CurrentPopulation["Positions"])\
        + C2*np.random.random()*(PastGlobalBest[0]-CurrentPopulation["Positions"])

        # position update
        # update position as position  + new velocity
        # then update the current popoilation with new function evaluation, Constraint violation
        CurrentPopulation["Positions"] = CurrentPopulation["Positions"] + CurrentPopulation["Velocities"]
        CurrentPopulation["FunctionValues"] = singleObjFuncs.Rastrigin(CurrentPopulation["Positions"])
        CurrentPopulation["ConstraintViolation"] = singleObjFuncs.constraintViolation(CurrentPopulation["Positions"])

        iteration = iteration + 1

        
    GlobalBestList.append(min(GlobalBestObjFuncValueList))
    
    BestParametersList.append(GlobalBestParameterList[GlobalBestObjFuncValueList.index(min(GlobalBestObjFuncValueList))]) 
# print("GlobalBestList",GlobalBestList)
# print("BestParametersList",BestParametersList)

mean = statistics.mean(GlobalBestList)
print(mean)
std = statistics.stdev(GlobalBestList)
print(std)
best = min(GlobalBestList)
print(best)
bestParameters = BestParametersList[GlobalBestList.index(min(GlobalBestList))]
print(bestParameters)

figF, axF = plt.subplots()
axF.set_title('Function Values')
axF.boxplot(GlobalBestList,showfliers=False)

plt.waitforbuttonpress(5000)