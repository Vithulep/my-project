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

def generate_next_list(l,cap):
    N = len(l)
    
def generate_next_alloc(new_allocation):
    flag1 = 1
    return new_allocation,flag1

def main():
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
    profit_max = 0
    #---------------Code for grid search---------------
    flag1 = 1 
    iter_no = 0
    while(flag1):
        #-------Running the old optimization algorithm-------
        new_allocation,flag1 =  generate_next_alloc(new_allocation,iter_no)
        iter_no += 1
        allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)
        for district in districts.values():
            district.produce(allocation,crops)
        transportation = Greedy(districts,crops)
        flag,feasibility = transportation.solution_feasible(districts,crops)
        if flag:
            transportation.start_transportation(districts,crops)
            total_revenue=0
            total_production_cost=0
            if transportation.is_solution(districts,crops):
                for district in districts.values():
                    total_revenue+=district.get_revenue(crops)
                    total_production_cost+=district.production_cost
                    profit = total_revenue-total_production_cost-transportation.cost
                    if profit > profit_max:
                        profit_max = profit
    et = time.time()
    # get execution time in minutes
    res = et - st
    final_res = res / 60
    print('Execution time:', final_res, 'minutes')




#-----------------------OLD GRID SEARCH which does not take in account the sum of land allocation------------------------
def main1():
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
    
    alloc1 = []
    profit1 = []
    sample = open('samplefile.txt', 'w')
    #---------------Code for grid search---------------
    no_rows,no_cols = len(districts.values()),len(list(crops.keys()))
    increments=[1]
    arr = [1 for i in range(no_rows*no_cols)] #initialize array with 1s
    index = [0 for i in range(no_rows*no_cols)]
    increments=[1]
    N=12
    for i in range(1,12,1):
        if i%2 != 0:
            increments.append(increments[i-1]*5)
        else:
            increments.append(increments[i-1]*2)
    ctr = 1
    pos = 0
    while(True):
        if ctr == (N)**(no_rows*no_cols):
            break
        if arr[pos]<increments[-1]:
            index[pos]+=1
            arr[pos]=increments[index[pos]]
            pos=0
            ctr += 1
            #Allocating values to districts
            allocator=0
            for district in districts.values():
                for crps in list(crops.keys()):
                    new_allocation.allocation[district.id][crps]=arr[allocator]
                    allocator+=1
        elif pos == (no_rows*no_cols-1) and arr[pos]<increments[-1]:
            arr[pos]=increments[index[pos]]
            index[pos]+=1
            pos=0
        elif pos == (no_rows*no_cols-1) and arr[pos]==increments[-1]: 
            pos=0
        elif arr[pos] == increments[-1]:
            index[pos]=0
            arr[pos]=increments[index[pos]]
            pos=pos+1 
        #-------Running the old optimization algorithm-------
        allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)
        temp = copy.deepcopy(new_allocation.allocation)
        alloc1.append(temp)
        for district in districts.values():
            district.produce(allocation,crops)
        transportation = Greedy(districts,crops)
        flag,feasibility = transportation.solution_feasible(districts,crops)
        if flag:
            transportation.start_transportation(districts,crops)
            total_revenue=0
            total_production_cost=0
            if transportation.is_solution(districts,crops):
                for district in districts.values():
                    total_revenue+=district.get_revenue(crops)
                    total_production_cost+=district.production_cost
            # alloc2=new_allocation.allocation
            # profit2=total_revenue-total_production_cost-transportation.cost
            # if profit2 not in profit1:
            profit1.append(total_revenue-total_production_cost-transportation.cost)
            # alloc1.append(new_allocation.allocation)
    # list1, list2 = zip(*sorted(zip(profit1, alloc1),reverse=True))
    # res = "\n".join("Profit:{} for allocation {}".format(x, y) for x, y in zip(list1, list2))
    # print(res, file = sample)
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
    sample.close()
    et = time.time()
    # get execution time in minutes
    res = et - st
    final_res = res / 60
    print('Execution time:', final_res, 'minutes')
    


if __name__ == '__main__':
    main()
