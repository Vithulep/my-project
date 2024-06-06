import os
import os.path as osp
import copy
import random
import argparse
import readfile
from allocation import Allocation
from transportation import Greedy
import time
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import csv

def parse_args():
    arg_parser = argparse.ArgumentParser()

    # input argument options
    arg_parser.add_argument(dest='example_path',
                            type=str,
                            help='Pass the path to the data folder')
    args = arg_parser.parse_args()
    return args

def main():
    # get the start time
    st = time.time()
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'

    districts=readfile.read_districts(example_path,'district.csv')
    crops = readfile.read_crops(example_path,'crop.csv')
    new_allocation = readfile.read_allocation(example_path,'new_allocation.csv',crops)
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)
    #Set the following parameters:
    compliance=readfile.read_parameters(example_path,'parameters.csv')
    alloc1 = []
    profit1 = []

    filename = "records.csv"
    #---------------Code for grid search---------------
    intitial_alloc1 = new_allocation.allocation['17']['maize']
    intitial_alloc2 = new_allocation.allocation['25']['tur']
    N = 10
    arr = [i for i in range(0,int(intitial_alloc1*N),int(intitial_alloc1*N/100))]
    arr2 = [i for i in range(0,int(intitial_alloc2*N),int(intitial_alloc2*N/100))]
    for (i,j) in zip(arr,arr2):
        # for district in districts.values():
        #     for crps in list(crops.keys()):
        #         if crps == 'rice' and district.id == '1':
        new_allocation.allocation['17']['maize']=i
        new_allocation.allocation['25']['tur']=j
        #-------Running the old optimization algorithm-------
        allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)
        temp = copy.deepcopy(new_allocation.allocation)
        alloc1.append(temp)
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
        profit1.append(total_revenue-total_production_cost-transportation.cost)

    # x=np.array(arr)
    # y=np.array(profit1)/1e6
    # X_Y_Spline = make_interp_spline(x, y)
 
    # # Returns evenly spaced numbers
    # # over a specified interval.
    # X_ = np.linspace(x.min(), x.max(), 500)
    # Y_ = X_Y_Spline(X_)
    
    # Plotting the Graph
    plt.plot(profit1)
    # plt.title("Varying rice allocation")
    # plt.xlabel("Rice allocation in hectares")
    # plt.ylabel("Profit in million rupees")
    # plt.ylim(0,9)
    plt.show()
    
    et = time.time()
    # get execution time in minutes
    res = et - st
    final_res = res / 60
    print('Execution time:', final_res, 'minutes')




if __name__ == '__main__':
    main()
