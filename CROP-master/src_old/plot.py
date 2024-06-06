import os
import os.path as osp
import argparse
import readfile
from allocation import Allocation
from transportation import Greedy
import matplotlib.pyplot as plt
import numpy as np
import csv


def calc_distances(districts):
    distances={}
    for d1 in districts.values():
        for d2 in districts.values():
            if d1.id != d2.id:
                distances[(d1.id,d2.id)]=d1.coordinates.distance(d2.coordinates)
    distances = dict(sorted(distances.items(), key=lambda item: item[1]))
    return distances


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

def parse_args():
    arg_parser = argparse.ArgumentParser()

    # input argument options
    arg_parser.add_argument(dest='example_path',
                            type=str,
                            help='Pass the path to the data folder')
    args = arg_parser.parse_args()
    return args

def plot_results(name, avg_dict):
    for state in avg_dict.keys():
        for i in range(len(avg_dict[state])):
            avg_dict[state][i]=avg_dict[state][i]/1e9
        plt.plot(avg_dict[state])

    # plt.title(name)
    plt.legend(list(avg_dict.keys()), shadow=True)
    plt.ylabel('In billion INR')
    plt.xlabel('Compliance')
    plt.xticks(list(range(11)),map(lambda x: x/10,list(range(11))))
    # plt.ylim(0,90)
    plt.show()

def compute_solution(example_path,districts,crops,allocation,distances,a=1):
    #--------------------FOR MAX DEMAND--------------------------
    # for crop in crops.keys():
    #     for i in crops[crop].get_max_demand().keys():
    #         crops[crop].get_max_demand()[i] *= a
    #--------------------For min demand---------------------------
    for crop in crops.keys():
        for i in crops[crop].get_min_demand().keys():
            crops[crop].get_min_demand()[i] = crops[crop].get_min_demand()[i]*a
    
    for district in districts.values():
        district.produce(allocation,crops)

    transportation = Greedy(districts,crops,distances)
    flag,feasibility = transportation.solution_feasible(districts,crops)
    if not flag:
        return 0,0,0

    transportation.start_transportation(districts,crops)
    total_revenue=0
    total_production_cost=0
    for district in districts.values():
        total_revenue+=district.get_revenue(crops)
        total_production_cost+=district.production_cost
    #--------------------------For max demand------------------------
    # for crop in crops.keys():
    #     for i in crops[crop].get_max_demand().keys():
    #         crops[crop].get_max_demand()[i] /= a
    #-------------------------For min demand-------------------------
    for crop in crops.keys():
        for i in crops[crop].get_min_demand().keys():
            crops[crop].get_min_demand()[i] = crops[crop].get_min_demand()[i]/a
    
    return total_revenue,total_production_cost,transportation.cost
   
    
# def main1():
#     args = parse_args()
#     example_path = args.example_path
#     if example_path[-1]!='/':
#         example_path+='/'

#     districts=readfile.read_districts(example_path,'district.csv')
#     crops = readfile.read_crops(example_path,'crop.csv')
#     new_allocation = readfile.read_allocation(example_path,'70_120_O_random.csv',crops)
#     standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)

#     # d={'Revenue':[],'Production Cost':[],'Transportation':[],'Net Income':[]}
#     d={'Revenue':[],'Production Cost':[],'Profit':[]}#,'Transportation':[]}


#     for i in range(11):
#         compliance=i/10
#         allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)
#         r,p,t=compute_solution(example_path,districts,crops,allocation)
#         d['Revenue'].append(r)
#         d['Production Cost'].append(p)
#         # d['Transportation'].append(t)
#         d['Profit'].append(r-p-t)

#     plot_results("Cost & Profit vs Compliance",d)

def main():
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'

    districts=readfile.read_districts(example_path,'district.csv')
    crops = readfile.read_crops(example_path,'crop.csv')
    new_allocation_11 = readfile.read_allocation(example_path,'out_RV_grid.csv',crops)
    # new_allocation_12 = readfile.read_allocation(example_path,'70_120_O_random.csv',crops)
    # new_allocation_13 = readfile.read_allocation(example_path,'80_120_crop_alloc.csv',crops)
    # new_allocation_14 = readfile.read_allocation(example_path,'80_140_crop_alloc.csv',crops)
    # new_allocation_15 = readfile.read_allocation(example_path,'1.5_7_crop_alloc.csv',crops)
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)
    d_11 = {'Revenue':[],'Production Cost':[],'Net Income':[],'Transportation':[]}
    # d_12 = {'Revenue':[],'Production Cost':[],'Net Income':[]}
    # d_13 = {'Revenue':[],'Production Cost':[],'Net Income':[]}
    # d_14 = {'Revenue':[],'Production Cost':[],'Net Income':[]}
    # d_15 = {'Revenue':[],'Production Cost':[],'Net Income':[]}
    distances = calc_distances(districts)
    x=[]
    for i in range(0,11,1):
        compliance=i/10
        x.append(compliance)
        allocation=Allocation(standard_allocation.allocation,new_allocation_11.allocation,compliance)
        r,p,t=compute_solution(example_path,districts,crops,allocation,distances,0.65/0.7)
        d_11['Revenue'].append(r)
        d_11['Production Cost'].append(p)
        d_11['Transportation'].append(t)
        d_11['Net Income'].append(r-p-t)

        
        # allocation=Allocation(standard_allocation.allocation,new_allocation_12.allocation,compliance)
        # r,p,t=compute_solution(example_path,districts,crops,allocation,0.69/0.7)
        # d_12['Revenue'].append(r)
        # d_12['Production Cost'].append(p)
        # d['Transportation'].append(t)
        # d_12['Net Income'].append(r-p-t)

        # allocation=Allocation(standard_allocation.allocation,new_allocation_13.allocation,compliance)
        # r,p,t=compute_solution(example_path,districts,crops,allocation,0.73/0.7)
        # d_13['Revenue'].append(r)
        # d_13['Production Cost'].append(p)
        # # d['Transportation'].append(t)
        # d_13['Net Income'].append(r-p-t)
    
    # for i in range(len(d_11['Revenue'])):
    #     d_11['Revenue'][i] = d_11['Revenue'][i]/1e9
    #     d_12['Revenue'][i] = d_12['Revenue'][i]/1e9
    #     d_13['Revenue'][i] = d_13['Revenue'][i]/1e9
    plt.plot(x,d_11['Revenue'], color='r', label='110 % of production')
    # plt.plot(x,d_12['Revenue'], color='g', label='120 % of production')
    # plt.plot(x,d_13['Revenue'], color='b', label='130 % of production')
    # # plt.title('Revenue vs compliance for various min_demand')
    # plt.ylabel('Revenue in billion INR')
    # print(len(d_11['Net Income']))
    #-------------------------for plotting the ghraph------------------------------------
    for i in range(len(d_11['Net Income'])):
        d_11['Net Income'][i] = d_11['Net Income'][i]/1e9
        # d_12['Net Income'][i] = d_12['Net Income'][i]/1e9
        # d_13['Net Income'][i] = d_13['Net Income'][i]/1e9
    plt.plot(x[3:],d_11['Net Income'][3:], color='r', label='60 % of production')
    # plt.plot(x[3:],d_12['Net Income'][3:], color='g', label='70 % of production')
    # plt.plot(x[3:],d_13['Net Income'][3:], color='b', label='80 % of production')
    # plt.title('Profit vs compliance for various min_demand')
    plt.ylabel('Profit in billion INR')
    plt.xlabel('Compliance')
    plt.legend()
    x_ticks = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
    plt.xticks(x[3:],labels=x_ticks[3:])
    plt.ylim(200,240)
    plt.show()



main()
