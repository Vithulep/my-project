import os
import os.path as osp
import copy
import random
import argparse
import readfile
import math
from allocation import Allocation
from transportation import Greedy
from SimulatedAnnealing import SimulatedAnnealing
import time

def parse_args():
    arg_parser = argparse.ArgumentParser()

    # input argument options
    arg_parser.add_argument(dest='example_path',
                            type=str,
                            help='Pass the path to the data folder')
    args = arg_parser.parse_args()
    return args

def initialize_alloc(alloc,no_crops,no_dist,districts,crops,max_alloc):
    # print(no_crops,no_dist)
    for district,i in zip(districts.values(),range(no_dist)):  
        # print(district.id)
        temp=[]
        for crps in list(crops.keys()):
            temp.append(random.uniform(0,1))
        sum1=sum(temp)
        if sum1 != 0:
            for crps,j in zip(list(crops.keys()),range(no_crops)):
                # print(district.id, crps)
                alloc.allocation[district.id][crps] = (temp[j]*max_alloc[i])/sum1
    return alloc


def neighbour_alloc(alloc,no_crops,no_dist,districts,crops,max_alloc):
    fraction_to_move = 10
    #Converting to range 0-1 and adding randomness for movement
    test_list=[-1,1]
    for district,i in zip(districts.values(),range(no_dist)):
        temp=[]
        sum1=0
        for crps in list(crops.keys()):
            sum1 += alloc.allocation[district.id][crps]
        #converiting numbers to [0-1] range
        if sum1 != 0:
            for crps,j in zip(list(crops.keys()),range(no_crops)):
                alloc.allocation[district.id][crps] = alloc.allocation[district.id][crps]/sum1
                #Adding small noise [0-1]/100
                temp.append(alloc.allocation[district.id][crps])
                temp[j] = temp[j]+random.choice(test_list)*random.uniform(0,1)/fraction_to_move
                while not (temp[j]>=0 and temp[j]<=1):
                    temp[j] = alloc.allocation[district.id][crps] 
                    temp[j] = temp[j]+random.choice(test_list)*random.uniform(0,1)/fraction_to_move
                alloc.allocation[district.id][crps] = temp[j]
        #converting back to required range
        sum1=0
        for crps in list(crops.keys()):
            sum1 += alloc.allocation[district.id][crps]
        if sum1 !=0:
            for crps,j in zip(list(crops.keys()),range(no_crops)):
                alloc.allocation[district.id][crps] = (temp[j]*max_alloc[i])/sum1
    return alloc

def cost(standard_allocation,new_allocation,compliance,districts,crops):
    #------------------------------Running the old optimization algorithm---------------------------
    allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)
    for district in districts.values():
        district.produce(allocation,crops)
    transportation = Greedy(districts,crops)
    flag,feasibility = transportation.solution_feasible(districts,crops)
    total_revenue=0
    total_production_cost=0
    if flag:
        transportation.start_transportation(districts,crops)
        if transportation.is_solution(districts,crops):
            for district in districts.values():
                total_revenue+=district.get_revenue(crops)
                total_production_cost+=district.production_cost
    profit = total_revenue-total_production_cost-transportation.cost
    return profit,flag
 
def main():
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'


    districts=readfile.read_districts(example_path,'district.csv')
    crops = readfile.read_crops(example_path,'crop.csv')
    new_allocation = readfile.read_allocation(example_path,'new_allocation.csv',crops)
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)
    # print("hi")
    #Set the following parameters:
    compliance=readfile.read_parameters(example_path,'parameters.csv')
    ans1 = []
    ans2 = []
    i,j=0,0
    while i<10 and j<100:
        no_dist,no_crops = len(districts.values()),len(list(crops.keys()))
        max_alloc = []
        for district in districts.values():
            sum1=0
            for crps in list(crops.keys()):
                sum1 += new_allocation.allocation[district.id][crps]
            max_alloc.append(sum1)

        flag = False
        k=0
        while not flag and k <10:
            alloc = initialize_alloc(new_allocation,no_crops,no_dist,districts,crops,max_alloc)
            # print(alloc.allocation)
            profit,flag = cost(standard_allocation,alloc,compliance,districts,crops)
            # print(flag)
            k += 1 
        # print(profit,alloc.allocation)
        if k == 10:
            print("could not initialize")
            # return 0

        else:
            # SA = SimulatedAnnealing(alloc,cost,abs(profit)*10,(abs(profit)/1000),'slowDecrease',neighbour_alloc,no_crops,no_dist,districts,crops,max_alloc,compliance)
            SA = SimulatedAnnealing(alloc,cost,10**12,10**7,'slowDecrease',neighbour_alloc,no_crops,no_dist,districts,crops,max_alloc,compliance)
            alloc1 = SA.run()
            profit1,flag = cost(standard_allocation,alloc1,compliance,districts,crops)
            # print(profit1)
            if flag == True:
                i=i+1
                print('SA iteration: ',i)
                ans1.append(math.floor(profit1))
                ans2.append(copy.deepcopy(alloc.allocation))
            j+=1
        print(ans1)
        # index = ans1.index(max(ans1))
        # print(ans1[index])
        # print(ans2[index])
    
if __name__ == '__main__':
    main()

