import os
import os.path as osp
import copy
import random
import argparse
import readfile
from allocation import Allocation
from transportation import Greedy
import time

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

    # print("Inputs ")
    # print('Districts : ', end =" ")
    # for district in districts.values():
    #     print('[',district.id,':',district.name,']',end =" ")
#    print('Districts : ',list(districts.keys()))
    # print('\nCrops : ',list(crops.keys()))
    
    # N = 100 #grid search size
    alloc1 = []
    profit1 = []
    sample = open('samplefile.txt', 'w')
    #---------------Code for grid search---------------
    no_rows,no_cols = len(districts.values()),len(list(crops.keys()))
    min1,max1=0,10
    inc=1
    arr = [min1 for i in range(no_rows*no_cols)]
    ctr = 1
    pos = 0
    # temp_idx=0
    while(True):
        if ctr == (max1-min1+1)**(no_rows*no_cols):
            break
        if arr[pos]<max1:
            arr[pos]+=inc
            pos=0
            ctr += 1
            allocator=0
            for district in districts.values():
                for crps in list(crops.keys()):
                    new_allocation.allocation[district.id][crps]=arr[allocator]
                    allocator+=1
        elif pos == (no_rows*no_cols-1) and arr[pos]<max1:
            arr[pos]+=inc
            pos=0
        elif pos == (no_rows*no_cols-1) and arr[pos]==max1: 
            pos=0
        elif arr[pos] == max1:
            arr[pos]=min1
            pos=pos+1
        #-------Running the old optimization algorithm-------
        allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)
        # temp=new_allocation.allocation.copy()
        temp = copy.deepcopy(new_allocation.allocation)
        alloc1.append(temp)
        # print(alloc1[temp_idx])
        # temp_idx+=1
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
        # print(f"{ctr}:{total_revenue-total_production_cost-transportation.cost}")
        profit1.append(total_revenue-total_production_cost-transportation.cost)
                # alloc1.append(new_allocation.allocation)
    print(profit1)
    li=[]
    for i in range(len(profit1)):
        li.append([profit1[i],i])
    li.sort(reverse=True)
    sort_index = []
 
    for x in li:
        sort_index.append(x[1])
    res=""
    counter1=100
    for i in sort_index:
        res = "".join("Profit:{} for allocation {}".format(profit1[i],alloc1[i]))
        print(res, file = sample)
        if counter1 == 0:
            break
        else:
            counter1=counter1-1
    # res = "\n".join("Profit:{} for allocation {}".format(x, y) for x, y in zip(profit1, alloc1))
    # list1, list2 = zip(*sorted(zip(profit1, alloc1),reverse=True))
    # res = "\n".join("Profit:{} for allocation {}".format(x, y) for x, y in zip(list1, list2))
    sample.close()
    # get the start time
    et = time.time()
    # get execution time in minutes
    res = et - st
    final_res = res / 60
    print('Execution time:', final_res, 'minutes')

    # for i in alloc1:
    #     print(i)
    # print(district.id,":",crps)
    # print('New allocatition : ',new_allocation.allocation)
    # print('Standard allocation : ',standard_allocation.allocation)
    # print("")
    # print("Stock after production")

    # allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)
    # for district in districts.values():
    #     district.produce(allocation,crops)
    #     print('District : ',district.name,', stock : ',district.stock)

    # print("")

    # transportation = Greedy(districts,crops)
    # flag,feasibility = transportation.solution_feasible(districts,crops)
    # if not flag:
    #     print('Solution not feasible : ',feasibility)
    #     return

    # print("Solution status : ",transportation.is_solution(districts,crops))
    # transportation.start_transportation(districts,crops)
    # print("Logs : ",transportation.logs)
    # print("")
    # print("Final Distribution : ")
    # for district in districts.values():
    #     print('District : ',district.name,', stock : ',district.stock)

    # print("")
    # print("Solution status : ",transportation.is_solution(districts,crops))

    # total_revenue=0
    # total_production_cost=0
    # for district in districts.values():
    #     total_revenue+=district.get_revenue(crops)
    #     total_production_cost+=district.production_cost

    # print("")
    # print("Total revenue from sale : ", round(total_revenue,2))
    # print("")

    # print("Total cost of production : ", round(total_production_cost,2))
    # print("")

    # print("Total transport cost : ", round(transportation.cost,2))
    # print("")




if __name__ == '__main__':
    main()
