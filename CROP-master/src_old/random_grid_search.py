import os
import os.path as osp
import copy
import random
import argparse
import readfile
from allocation import Allocation
from transportation import Greedy
from transportation import Greedy_price
from transportation import Bipartite_price
import numpy as np
import time
import csv
import itertools
import matplotlib.pyplot as plt
from geopy.distance import great_circle
from gurobipy import *

def solution_feasible(dist_crop_numpy,stock_numpy):
    total_stock = np.sum(stock_numpy,axis = 0)
    flag = True
    total_min_demand = np.zeros((stock_numpy.shape[1]))
    feasibility =  np.full((stock_numpy.shape[1]), True, dtype=bool)
    total_min_demand = [np.sum(dist_crop_numpy[i,:,6]) for i in range(dist_crop_numpy.shape[0])]
    for i in range(total_stock.shape[0]):
        if total_stock[i] < total_min_demand[i]:
            flag = False
            feasibility[i] = False
    return flag,feasibility

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

def calc_arrival_prices(districts,crops):
    all_crop_arrival_prices =  {}
    for crps,crop_id in zip(crops,crops.keys()):
        arrival_costs ={}
        for d1 in districts.values():
            for d2 in districts.values():
                    if d1.id != d2.id:
                        # price from d1->d2
                        arrival_costs[(d1.id,d2.id)]=d1.coordinates.distance(d2.coordinates)*crops[crps].transport_cost+d2.get_price_per_ton(crops,crop_id)
        arrival_costs = dict(sorted(arrival_costs.items(), key=lambda item: item[1]))
        all_crop_arrival_prices[crps.lower()] = arrival_costs
    return all_crop_arrival_prices

def calc_distances(districts):
    distances={}
    for d1 in districts.values():
        for d2 in districts.values():
            if d1.id != d2.id:
                distances[(d1.id,d2.id)]=d1.coordinates.distance(d2.coordinates)
    distances = dict(sorted(distances.items(), key=lambda item: item[1]))
    return distances

# def calc_arrival_prices(districts,crops):
#     all_crop_arrival_prices =  {}
#     for crps,crop_id in zip(crops,crops.keys()):
#         for d1 in districts.values():
#             # print(d1.id)
#             # for crps,j in zip(crops,range(len(crops))):
#             for d2 in districts.values():
#                     if d1.id != d2.id:
#                         # price from d1->d2
#                         arrival_costs[(d1.id,d2.id)]=d1.coordinates.distance(d2.coordinates)*crops[crps].transport_cost+d2.get_price_per_ton(crops,crop_id)
#                         # print(self.arrival_costs[(d1.id,d2.id,crop_id)],d1.id,d2.id,crps,d1.coordinates.distance(d2.coordinates),crops[crps].transport_cost,d2.get_price_per_ton(crops,crop_id))
#         arrival_costs = dict(sorted(arrival_costs.items(), key=lambda item: item[1]))
#         all_crop_arrival_prices[crps] = arrival_costs
#     print(all_crop_arrival_prices)

def allocation_to_csv(allocation):
    data = ['District ID']
    index = [(i+1) for i in range(len(allocation.allocation.keys()))]
    for i in allocation.allocation.keys():
        data.append(allocation.allocation[i])
    df = pd.DataFrame(data, index)
    df.to_csv('out_RV_grid.csv')

def parse_args():
    arg_parser = argparse.ArgumentParser()

    # input argument options
    arg_parser.add_argument(dest='example_path',
                            type=str,
                            help='Pass the path to the data folder')
    args = arg_parser.parse_args()
    return args

#----------------------------------------Numpy-----------------------------------------
def main1():
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
    dist_crop_numpy = np.swapaxes(dist_crop_numpy,0,1) #dim_0->crop_no, dim_1->dist_no, dim_3->cost[2],yeild[3],price[4],max_demand[5],min_demand[6]
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
    
    No_iter = 1

    distances_numpy = calc_distances_numpy(district_numpy)
    print(distances_numpy)
    sum1 = np.zeros((no_dist,1))

    #---------------------------------COMMENTED FROM HERE--------------------------
    # for _ in range(No_iter):
    #     if _%10000==0:
    #         print(f"iteration no {i} ")
        
    #     new_allocation_numpy = np.random.uniform(0,1,(no_dist,no_crop))
    #     sum1 = np.sum(new_allocation_numpy, axis = 1).reshape(no_dist,1)
    #     new_allocation_numpy = new_allocation_numpy/sum1
    #     new_allocation_numpy = new_allocation_numpy*max_alloc
        
    #     # for j in range(no_dist):
    #     #     for k in range(no_crop):
    #             # new_allocation_numpy[j,k] = np.random.uniform(0,1)
    #     # for j in range(no_dist):
    #     #     sum1[j] = new_allocation_numpy[j,:].sum()
    #     # for j in range(no_dist):
    #     #     for k in range(no_crop):
    #     #         # print(max_alloc[j,0])
    #     #         # print(new_allocation_numpy[j,k])
    #     #         new_allocation_numpy[j,k] = new_allocation_numpy[j,k]*max_alloc[j]
    #     #         new_allocation_numpy[j,k] = new_allocation_numpy[j,k]/sum1[j]
    #         # print(new_allocation_numpy[0,0])
    # # #     # -------Running the old optimization algorithm-------
    #     allocation_numpy =  (1-compliance)*standard_allocation_numpy+compliance*new_allocation_numpy
    #     stock_numpy=np.zeros((len(district_numpy),len(crops_numpy)))              # Districts->rows, crops->colunms 
    #     production_cost=np.zeros((len(district_numpy),len(crops_numpy)))          # Districts->rows, crops->colunms
    #     for k in range(len(crops_numpy)):
    #         stock_numpy[:,k] = allocation_numpy[:,k]*dist_crop_numpy[k][:,3]
    #         production_cost[:,k] = allocation_numpy[:,k]*dist_crop_numpy[k][:,2]
    #      # checking feasibility
    #     flag,feasibility = solution_feasible(dist_crop_numpy, stock_numpy)
    #     if not flag:
    #         continue
    #     #Greedy transportation
    #     transportation_cost = 0

    #     for j in range(distances_numpy.shape[1]):
    #         for i in range(crops_numpy.shape[0]):
    #             d1_id = int(distances_numpy[1,j])        #reciver
    #             d2_id = int(distances_numpy[2,j])        #donor
    #             # checking if donor can give and reciver can recieve 
    #             if d1_id != d2_id:
    #                 is_satisfied = stock_numpy[d1_id,i] >= dist_crop_numpy[i,d1_id,6]
    #                 can_give = stock_numpy[d2_id,i] > dist_crop_numpy[i,d2_id,6]
    #                 if not (is_satisfied) and (can_give):
    #                     requirement = dist_crop_numpy[i,d1_id,6] - stock_numpy[d1_id,i]
    #                     amt_give = stock_numpy[d2_id,i] - dist_crop_numpy[i,d2_id,6]
    #                     transport_amt = min(requirement,amt_give)
    #                     stock_numpy[d1_id,i] = stock_numpy[d1_id,i] + transport_amt
    #                     stock_numpy[d2_id,i] = stock_numpy[d2_id,i] - transport_amt
    #                     transportation_cost += distances_numpy[0,j] * transport_amt * crops_numpy[i,1]
        
    #     flag,feasibility = solution_feasible(dist_crop_numpy, stock_numpy)
    #     # print(f'prod cost shape{production_cost.shape}')
    #     total_production_cost = np.sum(production_cost)
    #     total_revenue=0
    #     for i in range(dist_crop_numpy.shape[0]):   # for all crops
    #         for j in range(stock_numpy.shape[0]):   # for all districts 
    #             total_revenue += max(stock_numpy[j,i],dist_crop_numpy[i,j,5])*dist_crop_numpy[i,j,4]*10

    #     profit1 = total_revenue-total_production_cost-transportation_cost
    #     if profit1 > best_profit:
    #         best_profit = profit1
    # # print(best_profit)
    # et = time.time()
    # res = et - st
    # final_res = res / 60
    # print('Execution time:', final_res, 'minutes')

#---------------------------------With classes and sting--------------------------------
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
    compliance=readfile.read_parameters(example_path,'parameters.csv')

    best_profit = 0
    best_revenue =  0
    best_COP = 0
    best_transport_cost =  0
    # best_alloc = copy.deepcopy(new_allocation.allocation)
    #---------------Code for grid search---------------
    no_dist= len(districts.values())
    max_alloc = []


    for district in districts.values():
        sum1=0
        for crps in list(crops.keys()):
            sum1 += new_allocation.allocation[district.id][crps]
        max_alloc.append(sum1)
    
    No_iter = 10000000

        
    distances = calc_distances(districts)
    print('Greedy')

    # all_crop_arrival_prices = calc_arrival_prices(districts,crops)
    # print('Greedy_price')

    # distances = calc_distances(districts)
    # print('Bipartite')

    for _ in range(No_iter):
        if _%10000==0:
            print(f"iteration no {_} ")
        for district,j in zip(districts.values(),range(no_dist)):
            sum1=0
            for crps in list(crops.keys()):
                new_allocation.allocation[district.id][crps]=random.uniform(0,1)
                sum1+=new_allocation.allocation[district.id][crps]
            for crps in list(crops.keys()):
                new_allocation.allocation[district.id][crps]=(new_allocation.allocation[district.id][crps]/sum1)*max_alloc[j]

        # -------Running the old optimization algorithm-------
        allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)
        for district in districts.values():
            district.produce(allocation,crops)
        
        transportation = Greedy(districts,crops,distances)

        # transportation = Greedy_price(districts,crops,all_crop_arrival_prices)

        # transportation = Bipartite_price(districts,crops,distances)

        flag,feasibility = transportation.solution_feasible(districts,crops)
        total_revenue=0
        total_production_cost=0
        if flag:
            transportation.start_transportation(districts,crops)
            if transportation.is_solution(districts,crops):
                for district in districts.values():
                    total_revenue+=district.get_revenue(crops)
                    total_production_cost+=district.production_cost
        profit1 = total_revenue-total_production_cost-transportation.cost
        if profit1 > best_profit:
            best_profit = profit1
            best_revenue =  total_revenue
            best_COP = total_production_cost
            best_transport_cost =  transportation.cost
            # best_alloc = copy.deepcopy(new_allocation.allocation)
    print(f'Profit = {best_profit}')
    print(f'Revenue = {best_revenue}')
    print(f'COP = {best_COP}')
    print(f'Transport cost = {best_transport_cost}')
    # print(best_alloc)
    et = time.time()
    res = et - st
    final_res = res / 60
    print('Execution time:', final_res, 'minutes')

#-----------------------------------------Old code---------------------------------------------------
def main2():
    # get the start time
    st = time.time()
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'

    districts=readfile.read_districts(example_path,'district.csv')
    #Changed demand in readfile
    crops = readfile.read_crops(example_path,'crop.csv')
    new_allocation = readfile.read_allocation(example_path,'new_allocation.csv',crops)
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)
    compliance=readfile.read_parameters(example_path,'parameters.csv')
    
    profit1 = []
    alloc1 = []
    # sample = open('samplefile.txt', 'w')

    #---------------Code for grid search---------------
    no_dist= len(districts.values())
    max_alloc = []

    for district in districts.values():
        sum1=0
        for crps in list(crops.keys()):
            sum1 += new_allocation.allocation[district.id][crps]
        max_alloc.append(sum1)
    
    No_iter = 1000000
    
    #for storing allocations for districts
    temp = []
    
    for i in range(No_iter):
        if i%10000==0:
            print(f"iteration no {i} for allocation")
        for district,j in zip(districts.values(),range(no_dist)):
            sum1=0
            for crps in list(crops.keys()):
                new_allocation.allocation[district.id][crps]=random.uniform(0,1)
                sum1+=new_allocation.allocation[district.id][crps]
            for crps in list(crops.keys()):
                new_allocation.allocation[district.id][crps]=(new_allocation.allocation[district.id][crps]/sum1)*max_alloc[j]
        temp.append(copy.deepcopy(new_allocation.allocation))


    for i in range(len(temp)):
        if i%1000==0:
            print(f'iteration:{i} for profit calculation')

        new_allocation.allocation=temp[i]

        # -------Running the old optimization algorithm-------
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
        profit1.append(total_revenue-total_production_cost-transportation.cost)
        alloc1.append(temp[i])
    print(len(list(filter(lambda x: (x > 0), profit1))))
    max_profit = max(profit1)
    idx = profit1.index(max_profit)
    print(max_profit)
    print(alloc1[idx])  
    # profit1.sort()
    # profit1=np.array(profit1)
    # plt.plot(profit1)
    # plt.title("Random grid search outputs")
    # plt.xlabel('Number of grid points')
    # plt.ylabel('Profit in rupees') 
    # plt.show()

    # print(alloc1[idx].keys())
    # print(districts.keys())
    
    # fields = []
    # fields.append('CROP')
    # for i in alloc1[idx].keys():
    #     fields.append(i)
    # print(fields)

    # fields = [ 'org', '2015', '2014', '2013' ]
    # dw     = alloc1[idx]

    # with open("test_output.csv", "w") as f:
    #     w = csv.DictWriter(f, fields)
    #     w.writeheader()
        # for k in dw:
        #     w.writerow({field: dw[k].get(field) or k for field in fields})
    # with open("test_output.csv", "w") as f:
    #     w = csv.DictWriter(f, fields)
    #     w.writeheader()
    #     for k in alloc1[idx]:
    #         w.writerow({field: alloc1[idx][k].get(field) or k for field in fields})

    # plt.hist(profit1, 20)
    # plt.show()
        # print(temp, file = sample)
        # print('\n', file = sample)



    # i=0
    # for district in districts.values():
    #             for crps in list(crops.keys()):
    #                 new_allocation.allocation[district.id][crps]=decoded[i]
    #                 i+=1
    # print(f"Allocation: {new_allocation.allocation}")
    # allocation_to_csv(new_allocation)
    # sample.close()
    # # get the start time
    et = time.time()
    # get execution time in minutes
    res = et - st
    final_res = res / 60
    print('Execution time:', final_res, 'minutes')

if __name__ == '__main__':
    main()
