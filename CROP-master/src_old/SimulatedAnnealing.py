"""
Simulated Annealing Class
"""
import random
import math
# import main

class SimulatedAnnealing:
    def __init__(self, initialSolution, solutionEvaluator, initialTemp, finalTemp, tempReduction, neighborOperator, no_crops, no_dist, districts, crops, max_alloc, compliance,iterationPerTemp=100, alpha=10, beta=5):
        self.solution = initialSolution
        self.evaluate = solutionEvaluator
        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        self.no_crops = no_crops
        self.no_dist = no_dist
        self.districts = districts
        self.crops = crops
        self.max_alloc = max_alloc
        self.compliance = compliance
        self.iterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.beta = beta
        self.neighborOperator = neighborOperator

        if tempReduction == "linear":
            self.decrementRule = self.linearTempReduction
        elif tempReduction == "geometric":
            self.decrementRule = self.geometricTempReduction
        elif tempReduction == "slowDecrease":
            self.decrementRule = self.slowDecreaseTempReduction
        else:
            self.decrementRule = tempReduction

    def linearTempReduction(self):
        self.currTemp -= self.alpha

    def geometricTempReduction(self):
        self.currTemp *= self.alpha

    def slowDecreaseTempReduction(self):
        self.currTemp = self.currTemp / (1 + self.beta * self.currTemp)

    def isTerminationCriteriaMet(self):
        # can add more termination criteria
        return self.currTemp <= self.finalTemp #or self.neighborOperator(self.solution) == 0

    def run(self):
        ctr=0
        while not self.isTerminationCriteriaMet():
            ctr+=1
            # print('Running')
            # iterate that number of times
            for i in range(self.iterationPerTemp):
                # print('Iteration no of SA:',i)
                # # get all of the neighbors
                # neighbors = self.neighborOperator(self.solution)
                # pick a random neighbor
                newSolution = self.neighborOperator(self.solution,self.no_crops,self.no_dist,self.districts,self.crops,self.max_alloc)        #random.choice(neighbors)
                # get the cost between the two solutions
                # cost = self.evaluate(self.solution) - self.evaluate(newSolution)
                cost1 = self.evaluate(self.solution,newSolution,self.compliance,self.districts,self.crops)
                # if the new solution is better, accept it
                if cost1[0] >= 0:
                    self.solution = newSolution
                # if the new solution is not better, accept it with a probability of e^(-cost/temp)
                else:
                    # print(-cost1[0] / self.currTemp)
                    try:
                        if random.uniform(0, 1) < math.exp(-cost1[0] / self.currTemp):
                            self.solution = newSolution
                    except OverflowError as err:
                        print(-cost1[0], self.currTemp ,err)

            # decrement the temperature
            if ctr%2==0:
                self.decrementRule()
            return newSolution