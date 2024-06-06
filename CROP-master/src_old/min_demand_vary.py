import os
import os.path as osp
import argparse
import readfile
from allocation import Allocation
from import_export import Import_export
from transportation import Greedy
import matplotlib.pyplot as plt


def parse_args():
    arg_parser = argparse.ArgumentParser()

    # input argument options
    arg_parser.add_argument(dest='example_path',
                            type=str,
                            help='Pass the path to the data folder')
    args = arg_parser.parse_args()
    return args

def main():

    fasal = 'rice'
    zila = '8'
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'

    districts=readfile.read_districts(example_path,'district.csv')
    crops = readfile.read_crops(example_path,'crop.csv')
    new_allocation = readfile.read_allocation(example_path,'new_allocation.csv',crops)
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)
    import_export=readfile.read_import_export(example_path,'import_export.csv',crops)
    #Set the following parameters:
    compliance=readfile.read_parameters(example_path,'parameters.csv')


    allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)


    #Import_export
    import_export_obj = Import_export(allocation.allocation,import_export.allocation)
    import_export_obj.do_import_export()

    for district in districts.values():
        district.produce(allocation,crops)

    # initial_total_demand = 0
    # for fasal in crops.keys():
    #         for zila in districts.keys():
    #             initial_total_demand = crops[fasal].demand_min_dict[zila]
    #             initial_total_max_demand = crops[fasal].demand_max_dict[zila]
    #             crops[fasal].demand_min_dict[zila] = crops[fasal].demand_min_dict[zila]*0.8

    # flag1 = 0
    profit_list,revenue_list,cost_list,transportation_list= [],[],[],[]
    # total_demand_list = []
    
    # while flag1 == 0:
        
        # total_demand = 0
    # for fasal in crops.keys():
    #     for zila in districts.keys():
            # if crops[fasal].demand_max_dict[zila] < (crops[fasal].demand_min_dict[zila]+10):
                # crops[fasal].demand_min_dict[zila] = crops[fasal].demand_min_dict[zila]+10
                # if crops[fasal].demand_min_dict[zila] > crops[fasal].demand_max_dict[zila]:
                     
                # total_demand += crops[fasal].demand_min_dict[zila]
        
    # if initial_total_max_demand < total_demand:
    #         flag1 = 1 
    final_demand = crops[fasal].demand_max_dict[zila]
    initial_demand =  crops[fasal].demand_min_dict[zila]
    demand_vary =  [i for i in range(0,int(final_demand),int((final_demand)/100))]
    
    for i in demand_vary:
        crops[fasal].demand_min_dict[zila] = i
        transportation = Greedy(districts,crops)
        flag,feasibility = transportation.solution_feasible(districts,crops)
        if not flag:
            profit_list.append(0)
            revenue_list.append(0)
            cost_list.append(0)
            transportation_list.append(0)

        transportation.start_transportation(districts,crops)

        total_revenue=0
        total_production_cost=0
        for district in districts.values():
            total_revenue+=district.get_revenue(crops)
            total_production_cost+=district.production_cost
        revenue_list.append(total_revenue)
        cost_list.append(total_production_cost)
        transportation_list.append(transportation.cost)
        profit_list.append(round(total_revenue-total_production_cost-transportation.cost,2))

        # total_demand_list.append(total_demand)
    #plotting graphs

    x = demand_vary 
    plt.plot(x, profit_list,label = "Profit")
    plt.plot(x, cost_list,label = "Production cost")
    plt.plot(x, transportation_list,label = "Transportation cost")
    plt.plot(x, revenue_list,label = "Revenue")
    plt.axvline(x=initial_demand)
    plt.xlabel('In tonnes')
    # plt.ylabel('Profit')
    plt.legend()
    plt.title('Demand vs profit, revenue, cost')
    # plt.show()
    s=zila+' '+fasal+'.png'
    plt.savefig(s)
    plt.close()


def main2():

    fasal = 'rice'
    zila = '2'
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'

    districts=readfile.read_districts(example_path,'district.csv')
    crops = readfile.read_crops(example_path,'crop.csv')
    new_allocation = readfile.read_allocation(example_path,'new_allocation.csv',crops)
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)
    import_export=readfile.read_import_export(example_path,'import_export.csv',crops)
    #Set the following parameters:
    compliance=readfile.read_parameters(example_path,'parameters.csv')


    allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)


    #Import_export
    import_export_obj = Import_export(allocation.allocation,import_export.allocation)
    import_export_obj.do_import_export()

    for district in districts.values():
        district.produce(allocation,crops)

    initial_total_demand = 0
    for fasal in crops.keys():
            for zila in districts.keys():
                initial_total_demand = crops[fasal].demand_min_dict[zila]
                initial_total_max_demand = crops[fasal].demand_max_dict[zila]
                crops[fasal].demand_min_dict[zila] = crops[fasal].demand_min_dict[zila]*0.8

    flag1 = 0
    profit_list,revenue_list,cost_list,transportation_list= [],[],[],[]
    total_demand_list = []
    
    while flag1 == 0:
        
        total_demand = 0
        for fasal in crops.keys():
            for zila in districts.keys():
                if crops[fasal].demand_max_dict[zila] < (crops[fasal].demand_min_dict[zila]+10):
                    crops[fasal].demand_min_dict[zila] = crops[fasal].demand_min_dict[zila]+10
                    total_demand += crops[fasal].demand_min_dict[zila]
        
        if initial_total_max_demand < total_demand:
            flag1 = 1 
    # final_demand = crops[fasal].demand_max_dict[zila]

    # demand_vary =  [i for i in range(0,int(final_demand),int((final_demand)/100))]
    
    # for i in demand_vary:
    #     crops[fasal].demand_min_dict[zila] = i
        transportation = Greedy(districts,crops)
        flag,feasibility = transportation.solution_feasible(districts,crops)
        if not flag:
            profit_list.append(0)
            revenue_list.append(0)
            cost_list.append(0)
            transportation_list.append(0)

        transportation.start_transportation(districts,crops)

        total_revenue=0
        total_production_cost=0
        for district in districts.values():
            total_revenue+=district.get_revenue(crops)
            total_production_cost+=district.production_cost
        revenue_list.append(total_revenue)
        cost_list.append(total_production_cost)
        transportation_list.append(transportation.cost)
        profit_list.append(round(total_revenue-total_production_cost-transportation.cost,2))
        total_demand_list.append(total_demand)
    #plotting graphs

    x = total_demand_list 
    plt.plot(x, profit_list,label = "Profit")
    plt.plot(x, cost_list,label = "Production cost")
    plt.plot(x, transportation_list,label = "Transportation cost")
    plt.plot(x, revenue_list,label = "Revenue")
    plt.axvline(x=initial_total_demand)
    plt.xlabel('In tonnes')
    # plt.ylabel('Profit')
    plt.legend()
    plt.title('Demand vs profit, revenue, cost')
    # plt.show()
    plt.savefig('Demand.png')
    plt.close()




if __name__ == '__main__':
    main()

