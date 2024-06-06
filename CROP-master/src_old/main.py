import os
import os.path as osp
import argparse
import readfile
from allocation import Allocation
from import_export import Import_export
from transportation import Greedy_price
from transportation import Greedy
from transportation import Bipartite_price


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

def main():
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'

    districts=readfile.read_districts(example_path,'district.csv')
    
    crops = readfile.read_crops(example_path,'crop.csv')
    new_allocation = readfile.read_allocation(example_path,'new_allocation.csv',crops)
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)
    # import_export=readfile.read_import_export(example_path,'import_export.csv',crops)
    # Set the following parameters:
    compliance=readfile.read_parameters(example_path,'parameters.csv')

    # print("Inputs ")
    # print('Districts : ', end =" ")
    # for district in districts.values():
    #     print('[',district.id,':',district.name,']',end =" ")
    # print('Districts : ',list(districts.keys()))
    # print('Crops : ',list(crops.keys()))
    # for crps in list(crops.keys()):
    #     print(crops[crps].demand_min_dict)
    #     print(crops[crps].demand_min_dict)



    # print('New allocatition : ',new_allocation.allocation)
    # print('Standard allocation : ',standard_allocation.allocation)
    # # print('Import Export : ',import_export.allocation)
    # print("")
    # print("Stock after production")


    allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)

    # ############
    # # for district_id in allocation.allocation.keys():
    # #         for crop_id in allocation.allocation[district_id].keys():
    # #             print(import_export.allocation[district_id][crop_id])
    #             # pass
    # ############


    # #Import_export
    # # import_export_obj = Import_export(allocation.allocation,import_export.allocation)
    # # import_export_obj.do_import_export()

    for district in districts.values():
        district.produce(allocation,crops)
        # print('District : ',district.name,', stock : ',district.stock)

    distances = calc_distances(districts)
    transportation = Bipartite_price(districts,crops,distances)
    
    # all_crop_arrival_prices = calc_arrival_prices(districts,crops)
    # transportation = Greedy_price(districts,crops,all_crop_arrival_prices)


    # distances = calc_distances(districts)
    # transportation = Greedy(districts,crops,distances)


    flag,feasibility = transportation.solution_feasible(districts,crops)
    print(flag)
    if not flag:
        print('Solution not feasible : ',feasibility)
        return
    
    # flag,feasibility = transportation.upper_feasible(districts,crops)
    # print(flag)

    # print("Solution status : ",transportation.is_solution(districts,crops))
    transportation.start_transportation(districts,crops)
    # print("Logs : ",transportation.logs)
    # print("")
    # print("Final Distribution : ")
    # for district in districts.values():
    #     print('District : ',district.name,', stock : ',district.stock)

    # print("")
    print("Solution status : ",transportation.is_solution(districts,crops))

    total_revenue=0
    total_production_cost=0
    for district in districts.values():
        total_revenue+=district.get_revenue(crops)
        total_production_cost+=district.production_cost

    print("")
    print("Total revenue from sale : ", round(total_revenue,2))
    print("")

    print("Total cost of production : ", round(total_production_cost,2))
    print("")

    print("Total transport cost : ", round(transportation.cost,2))
    print("")

    print("Total profit : ", round(total_revenue-total_production_cost-transportation.cost,2))
    print("")





if __name__ == '__main__':
    main()
