import os
import os.path as osp
import argparse
import readfile
from allocation import Allocation
from transportation import Greedy_price
from transportation import Greedy
from transportation import Bipartite_price
import matplotlib.pyplot as plt
import numpy as np

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
        plt.plot(avg_dict[state])

    plt.title(name)
    plt.legend(list(avg_dict.keys()), loc='upper right', shadow=True)
    plt.ylabel('Cost')
    plt.xlabel('Compliance')
    plt.xticks(list(range(11)),map(lambda x: x/10,list(range(11))))
    plt.show()

def compute_solution(example_path,districts,crops,allocation):
    for district in districts.values():
        district.produce(allocation,crops)

    # distances = calc_distances(districts)
    # transportation = Bipartite_price(districts,crops,distances)
    
    # all_crop_arrival_prices = calc_arrival_prices(districts,crops)
    # transportation = Greedy_price(districts,crops,all_crop_arrival_prices)


    distances = calc_distances(districts)
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

    return total_revenue,total_production_cost,transportation.cost

def main():
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'

    districts=readfile.read_districts(example_path,'district.csv')
    crops = readfile.read_crops(example_path,'crop.csv')
    new_allocation = readfile.read_allocation(example_path,'allocation.csv',crops)
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)

    d={'Revenue':[],'Production Cost':[],'Transportation':[],'Net Income':[]}

    for i in range(11):
        compliance=i/10
        allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)
        r,p,t=compute_solution(example_path,districts,crops,allocation)
        d['Revenue'].append(r)
        d['Production Cost'].append(p)
        d['Transportation'].append(t)
        d['Net Income'].append(r-p-t)

    plot_results("Cost & Income vs compliance",d)

main()
