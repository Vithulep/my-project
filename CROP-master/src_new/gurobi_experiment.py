
import os
import os.path as osp
import argparse
import readfile
import numpy as np
import time
from geopy.distance import great_circle
from gurobipy import *
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    arg_parser = argparse.ArgumentParser()

    # input argument options
    arg_parser.add_argument(dest='example_path',
                            type=str,
                            help='Pass the path to the data folder')
    args = arg_parser.parse_args()
    return args

#---------------------------------------Gurobi-----------------------------------------
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
    dist_crop_numpy = [] #rows->districts, columns->dist_no,crop_no,crop_id,cost,yeild,price,max_demand,min_demand
    for district,j in zip(districts.values(),range(len(districts))):
        dist_crop_numpy.append(district.get_crop_details(crops,j))
    dist_crop_numpy = np.array(dist_crop_numpy)

    #-------------------At complince = 1---------------------------
    new_allocation = readfile.read_allocation(example_path,'allocation1.csv',crops)
    new_allocation_numpy = []               #rows -> districts, columns -> crops
    for i in new_allocation.allocation.keys():
        temp = []
        for j in crops.keys():
            temp.append(new_allocation.allocation[i][j])  
        new_allocation_numpy.append(temp)
    new_allocation_numpy = np.array(new_allocation_numpy)

    #------------------At complince = 0-----------------------
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)
    standard_allocation_numpy = []           #rows -> districts, columns -> crops
    for i in standard_allocation.allocation.keys():
        temp = []
        for j in crops.keys():
            temp.append(standard_allocation.allocation[i][j])
        standard_allocation_numpy.append(temp)
    standard_allocation_numpy = np.array(standard_allocation_numpy)
    no_dist= district_numpy.shape[0]
    no_crop = crops_numpy.shape[0]
    distances = np.zeros((no_dist,no_dist))
    cost = dist_crop_numpy[:,:,2]  #Cost per crop per ton
    crop_yield = dist_crop_numpy[:,:,3]     #Yeild tonnes per hectare
    price = dist_crop_numpy[:,:,4]      #price per quintal
    LD = dist_crop_numpy[:,:,6]*(0.7/0.9)         #Minimum demand of district in tonnes (0.7/0.9 for taking 70% of average demand)
    UD = dist_crop_numpy[:,:,5]         #Maximum dwmand of district in tonnes
    TC = crops_numpy[:,1]               #Transportation cost in rupees per ton per km
    # print(f'cost: \n {cost}')
    # print(f'crop_yied: \n{crop_yield}')
    # print(f'price: \n {price}')
    # print(f'LD: \n {LD}')
    # print(f'UD: \n {UD}')
    # print(f'TC: \n {TC}')
    
    
    #--------------Make changes here-------------------
    c=0
    allocation = (c/10)*standard_allocation_numpy + (1-c/10)*new_allocation_numpy
    for i in range(no_dist):
        for j in range(no_dist):
                if i == j:
                    # NEEDS TO BE UPDATED : Large value was taken to avoid transportation to self 
                    distances[i,j] = 10**5
                else: 
                    distances[i,j] = great_circle((district_numpy[i,2],district_numpy[i,1]),(district_numpy[j,2],district_numpy[j,1])).km
    #----------------------Gurobi code----------------------
    model = Model(name='CROP')
    #allocation = model.addVars(no_dist,no_crop,vtype=GRB.CONTINUOUS,lb = 0, name = 'allocation')  #For allocation
    x = model.addVars(no_dist,no_crop*no_dist,vtype=GRB.CONTINUOUS,lb = 0, name = 'x')  #For transportation 

    #----------------------------------Adding constarints-------------------------------------------------
    #Split these constraint equations 
    model.addConstrs((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i]\
                    for k in range(no_dist)))>=LD[i,j] for i in range(no_dist) for j in range(no_crop))

    model.addConstrs((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i]\
                    for k in range(no_dist)))<=UD[i,j] for i in range(no_dist) for j in range(no_crop))

    #model.addConstrs((sum(allocation[i,j] for j in range(no_crop))<=max_alloc[i]) for i in range(no_dist))

    COP = quicksum(allocation[i,j]*cost[i,j] for i in range(no_dist) for j in range(no_crop))
    Revenue = quicksum((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i] for k in range(no_dist)))*10*price[i,j] for i in range(no_dist) for j in range(no_crop))
    #Transporting crop j from district i -> k
    Transport_cost = quicksum((TC[j]*distances[i,k]*x[i,j*no_dist+k]) for k in range(no_dist) for i in range(no_dist) for j in range(no_crop))
    obj_fn =  Revenue - COP -  Transport_cost
    model.setObjective(obj_fn,GRB.MAXIMIZE)
    model.optimize()
    #----------------------------For printing the variable values-----------------------------------
    p = 0
    r = 0
    t = 0
    if model.status == GRB.Status.OPTIMAL:
        values = model.getAttr("X", model.getVars())
        values = np.array(values)
        #alloc = values[0:no_dist*no_crop].reshape((no_dist,no_crop))
        alloc=allocation
        transport_qty = values.reshape((no_dist,no_dist*no_crop))
        # print(f'Allocation: \n {alloc}')
        # print(f'Transported quantity: \n {transport_qty}')
        # np.savetxt('transport.csv', transport_qty, delimiter=',')
        stock = np.zeros(alloc.shape)               #Final Stock
        for i in range(no_dist):
            for j in range(no_crop):
                stock[i,j] = alloc[i,j]*crop_yield[i,j]-sum(transport_qty[i,j*no_dist+k] for k in range(no_dist))+sum(transport_qty[k,j*no_dist+i] for k in range(no_dist))
        # print(f'Stock: \n {stock}')
        p = sum(alloc[i,j]*cost[i,j] for i in range(no_dist) for j in range(no_crop))
        print(f"COP = {p}")
        r = sum((alloc[i,j]*crop_yield[i,j]-sum(transport_qty[i,j*no_dist+k] for k in range(no_dist))+sum(transport_qty[k,j*no_dist+i] for k in range(no_dist)))*10*price[i,j] for i in range(no_dist) for j in range(no_crop))
        print(f'Revenue ={r}')
        t = sum((TC[j]*distances[i,k]*transport_qty[i,j*no_dist+k]) for k in range(no_dist) for i in range(no_dist) for j in range(no_crop))
        print(f'Transport cost = {t}')
        print(f'Profit = {r-p-t}')


if __name__ == '__main__':
    main()
