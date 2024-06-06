import os
import os.path as osp
import copy
import random
import argparse
import readfile
from allocation import Allocation
from transportation import Greedy
import pandas as pd
import numpy as np
import time
import csv
import itertools
import matplotlib.pyplot as plt
from geopy.distance import great_circle
import numba
from numba import cuda

def is_solution(dist_crop_numpy,stock_numpy):
    for i in range(dist_crop_numpy.shape[0]):   # for all crops
        for j in range(stock_numpy.shape[0]):   # for all districts 
            if stock_numpy[j,i] < dist_crop_numpy[i,j,6]:
                return False
    return True

def calc_distances_numpy(district_numpy):
    N = district_numpy.shape[0]
    distances=np.zeros((3,N,N))
    temp1 = np.linspace(0,N-1,N)
    distances[2,:,:] = temp1
    temp2 = np.ones((N,N))
    for i in range(N):
        temp2[i,:] = i
    distances[1,:,:] = temp2
    for i in range(N):
        for j in range(N):
                distances[0,i,j]=great_circle((district_numpy[i,2],district_numpy[i,1]),(district_numpy[j,2],district_numpy[j,1])).km
    distances = np.reshape(distances,(3,N*N))
    b = distances.T
    b=b[np.argsort(b[:, 0])]
    b = b.T
    distances = b[:,N:]
    return distances            # 2D array: [[distances],[reciver],[donor]]


def parse_args():
    arg_parser = argparse.ArgumentParser()

    # input argument options
    arg_parser.add_argument(dest='example_path',
                            type=str,
                            help='Pass the path to the data folder')
    args = arg_parser.parse_args()
    return args

@cuda.jit
def run_on_gpu(results,no_iter,no_dist,no_crop,max_alloc,compliance,standard_allocation_numpy,district_numpy,crops_numpy,dist_crop_numpy,distances_numpy):
    
    i_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x

    for i in range(i_start, no_iter, threads_per_grid):
        # new_allocation_numpy = np.random.uniform(0,1,(no_dist,no_crop))
        for j in range(no_dist):
            for k in range(no_crop):
                new_allocation_numpy[j][k] = np.random.uniform(0,1)
        sum1 = np.sum(new_allocation_numpy, axis = 1).reshape(no_dist,1)
        new_allocation_numpy = new_allocation_numpy/sum1
        new_allocation_numpy = new_allocation_numpy*max_alloc
    # ----------------Running the old optimization algorithm-------------
        allocation_numpy =  (1-compliance)*standard_allocation_numpy+compliance*new_allocation_numpy
        stock_numpy=np.zeros((len(district_numpy),len(crops_numpy)))              # Districts->rows, crops->colunms 
        production_cost=np.zeros((len(district_numpy),len(crops_numpy)))          # Districts->rows, crops->colunms
        for i in range(len(crops_numpy)):
            stock_numpy[:,i] = allocation_numpy[:,i]*dist_crop_numpy[i][:,3]
            production_cost[:,i] = allocation_numpy[:,i]*dist_crop_numpy[i][:,2]
         # checking feasibility
            total_stock = np.sum(stock_numpy,axis = 0)
            flag = True
            total_min_demand = np.zeros((stock_numpy.shape[1]))
            feasibility =  np.full((stock_numpy.shape[1]), True, dtype=bool)
            total_min_demand = [np.sum(dist_crop_numpy[i,:,6]) for i in range(dist_crop_numpy.shape[0])]
            for i in range(total_stock.shape[0]):
                if total_stock[i] < total_min_demand[i]:
                    flag = False
                    feasibility[i] = False
        # flag,feasibility = solution_feasible(dist_crop_numpy, stock_numpy)
        if not flag:
            continue
        #Greedy transportation
        transportation_cost = 0

        for j in range(distances_numpy.shape[1]):
            for i in range(crops_numpy.shape[0]):
                d1_id = int(distances_numpy[1,j])        #reciver
                d2_id = int(distances_numpy[2,j])        #donor
                # checking if donor can give and reciver can recieve 
                if d1_id != d2_id:
                    is_satisfied = stock_numpy[d1_id,i] >= dist_crop_numpy[i,d1_id,6]
                    can_give = stock_numpy[d2_id,i] > dist_crop_numpy[i,d2_id,6]
                    if not (is_satisfied) and (can_give):
                        requirement = dist_crop_numpy[i,d1_id,6] - stock_numpy[d1_id,i]
                        amt_give = stock_numpy[d2_id,i] - dist_crop_numpy[i,d2_id,6]
                        transport_amt = min(requirement,amt_give)
                        stock_numpy[d1_id,i] = stock_numpy[d1_id,i] + transport_amt
                        stock_numpy[d2_id,i] = stock_numpy[d2_id,i] - transport_amt
                        transportation_cost += distances_numpy[0,j] * transport_amt * crops_numpy[i,1]
        
        # flag,feasibility = solution_feasible(dist_crop_numpy, stock_numpy)

        total_production_cost = np.sum(production_cost)
        total_revenue=0
        for i in range(dist_crop_numpy.shape[0]):   # for all crops
            for j in range(stock_numpy.shape[0]):   # for all districts 
                total_revenue += max(stock_numpy[j,i],dist_crop_numpy[i,j,5])*dist_crop_numpy[i,j,4]*10
        results[i] = total_revenue-total_production_cost-transportation_cost

#----------------------------------------Numpy-----------------------------------------
def main():
    # get the start time
    st = time.time()
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'

    #--------------------Districts------------------------------
    districts=readfile.read_districts(example_path,'district.csv')
    district_numpy = []
    for i in districts.keys():
        district_numpy.append([int(districts[i].id)-1,districts[i].coordinates.longitude,districts[i].coordinates.latitude])
    district_numpy = np.array(district_numpy)

    #----------------------Crops---------------------------------
    crops = readfile.read_crops(example_path,'crop.csv')
    crops_numpy = []
    for i,j in zip(crops.keys(),range(len(crops))):
        crops_numpy.append([j,crops[i].transport_cost])
    crops_numpy = np.array(crops_numpy)             #Rows->crops, columns-> transportation_costs
    dist_crop_numpy1 = [] #rows->districts, columns->dist_no,crop_no,crop_id,cost,yeild,price,max_demand,min_demand
    for district,j in zip(districts.values(),range(len(districts))):
        dist_crop_numpy1.append(district.get_crop_details(crops,j))
    dist_crop_numpy1 = np.array(dist_crop_numpy1)
    dist_crop_numpy1 = np.swapaxes(dist_crop_numpy1,0,1) #dim_0->crop_no, dim_1->dist_no, dim_3->cost[2],yeild[3],price[4],max_demand[5],min_demand[6]
    dist_crop_numpy = np.zeros_like(dist_crop_numpy1)
    dist_crop_numpy = dist_crop_numpy1
    #-------------------new_allocation---------------------------
    new_allocation = readfile.read_allocation(example_path,'new_allocation.csv',crops)
    new_allocation_numpy = []               #rows -> districts, columns -> crops
    for i in new_allocation.allocation.keys():
        temp = []
        for j in crops.keys():
            temp.append(new_allocation.allocation[i][j])  
        new_allocation_numpy.append(temp)
    new_allocation_numpy = np.array(new_allocation_numpy)

    #------------------standard_allocation-----------------------
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)
    standard_allocation_numpy = []           #rows -> districts, columns -> crops
    for i in standard_allocation.allocation.keys():
        temp = []
        for j in crops.keys():
            temp.append(standard_allocation.allocation[i][j])
        standard_allocation_numpy.append(temp)
    standard_allocation_numpy = np.array(standard_allocation_numpy)

    compliance=readfile.read_parameters(example_path,'parameters.csv')

    best_profit = 0
    #---------------Code for grid search---------------
    no_dist= standard_allocation_numpy.shape[0]
    no_crop = standard_allocation_numpy.shape[1]
    max_alloc = np.sum(new_allocation_numpy, axis = 1).reshape(no_dist,1) # 1D np array with 30 elements    
    
    # for j in range(no_dist):
    #     for k in range(no_crop):
    #         print(new_allocation_numpy[j,k])
    no_iter = 32 * 1000
    results_numpy = np.zeros(no_iter)
    distances_numpy = calc_distances_numpy(district_numpy)

    threads_per_block = 256
    blocks_per_grid_gs = 32 * no_iter//32
    
    results_numpy_dev = cuda.to_device(results_numpy) 
    no_iter_dev = cuda.to_device(no_iter)
    no_dist_dev = cuda.to_device(no_dist)
    no_crop_dev = cuda.to_device(no_crop)
    max_alloc_dev = cuda.to_device(max_alloc)
    compliance_dev = cuda.to_device(compliance)
    standard_allocation_numpy_dev = cuda.to_device(standard_allocation_numpy)
    district_numpy_dev = cuda.to_device(district_numpy)
    crops_numpy_dev = cuda.to_device(crops_numpy)
    dist_crop_numpy_dev = cuda.to_device(dist_crop_numpy)
    distances_numpy_dev = cuda.to_device(distances_numpy)

    run_on_gpu[blocks_per_grid_gs, threads_per_block](results_numpy_dev,no_iter_dev,no_dist_dev,no_crop_dev,max_alloc_dev,compliance_dev,standard_allocation_numpy_dev,district_numpy_dev,crops_numpy_dev,dist_crop_numpy_dev,distances_numpy_dev)

    results_numpy = results_numpy_dev.copy_to_host()
    
    best_profit = results_numpy.max()
    print(best_profit)
    et = time.time()
    res = et - st
    final_res = res / 60
    print('Execution time:', final_res, 'minutes')



if __name__ == '__main__':
    main()
