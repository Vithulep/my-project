import os
import os.path as osp
import argparse
import readfile
from allocation import Allocation
from import_export import Import_export
from transportation import Greedy
import matplotlib.pyplot as plt
import csv


def plot_histogram(x,s):
    plt.hist(x)
    plt.xlabel('Profit in Rupees')
    plt.ylabel('Frequency')
    plt.title(s)
    plt.show() 

def parse_args():
    arg_parser = argparse.ArgumentParser()

    # input argument options
    arg_parser.add_argument(dest='example_path',
                            type=str,
                            help='Pass the path to the data folder')
    args = arg_parser.parse_args()
    return args

#For getting images for all the crops
def main1():

    fasal = 'jowar'
    zila = '6'
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


    for fasal in crops.keys():
        for zila in districts.keys():
            # print(districts[zila].name)
            initial_price = crops[fasal].price_dict[zila]
            if initial_price < 10:
                break
            min_ratio,max_ratio = 0.1,10
            a=int(initial_price*min_ratio)
            b=int(initial_price*max_ratio)
            price_vary =  [i for i in range(a,b,int((b-a)/100))]
            
            profit_list,revenue_list,cost_list,transportation_list= [],[],[],[]

            for i in price_vary:
                crops[fasal].price_dict[zila] = i
                transportation = Greedy(districts,crops)
                flag,feasibility = transportation.solution_feasible(districts,crops)
                if not flag:
                    profit_list.append(0)

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
            #plotting graphs
            

            x = price_vary    
            plt.plot(x, profit_list,label = "Profit")
            plt.plot(x, cost_list,label = "Production cost")
            plt.plot(x, transportation_list,label = "Transportation cost")
            plt.plot(x, revenue_list,label = "Revenue")
            plt.axvline(x=initial_price)
            plt.xlabel('Price of maize in Bagalkot')
            plt.xlabel('In rupees')
            # plt.ylabel('Profit')
            plt.legend()
            plt.title('Price vs profit, revenue, cost')
            # plt.show()
            s=districts[zila].name+' '+fasal+' '+'.png'
            plt.savefig(s)
            plt.close()


#For running the model for actual prices 
def main():

    fasal = 'jowar'
    zila = '6'
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
    prices = []

    #Make changes here
    with open(example_path+'bidar_jowar.csv', mode ='r',encoding='utf-8-sig')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            # print(lines)
            prices.append(int(lines[0]))
    # print(prices)

    allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)


    #Import_export
    import_export_obj = Import_export(allocation.allocation,import_export.allocation)
    import_export_obj.do_import_export()

    for district in districts.values():
        district.produce(allocation,crops)

    initial_price = crops[fasal].price_dict[zila]
    
    profit_list,revenue_list,cost_list,transportation_list= [],[],[],[]

    for i in prices:
        crops[fasal].price_dict[zila] = i
        transportation = Greedy(districts,crops)
        flag,feasibility = transportation.solution_feasible(districts,crops)
        if not flag:
            profit_list.append(0)

        transportation.start_transportation(districts,crops)

        total_revenue=0
        total_production_cost=0
        for district in districts.values():
            total_revenue+=district.get_revenue(crops)
            total_production_cost+=district.production_cost
        # revenue_list.append(total_revenue)
        # cost_list.append(total_production_cost)
        # transportation_list.append(transportation.cost)
        profit_list.append(round(total_revenue-total_production_cost-transportation.cost,2))

    # print(profit_list)
    # #plotting graphs

    # x = prices    
    # plt.plot(x, profit_list,label = "Profit")
    # # plt.plot(x, cost_list,label = "Production cost")
    # # plt.plot(x, transportation_list,label = "Transportation cost")
    # # plt.plot(x, revenue_list,label = "Revenue")
    # plt.axvline(x=initial_price)
    # plt.xlabel(f'Price of {fasal} in {districts[zila].name}')
    # plt.xlabel('In rupees')
    # # plt.ylabel('Profit')
    # plt.legend()
    # plt.title('Price vs profit, revenue, cost')
    # # plt.show()
    # s=districts[zila].name+' '+fasal+' '+'.png'
    # plt.savefig(s)
    # plt.close()


    plot_histogram(profit_list,f'Profit distribution of {fasal} in {districts[zila].name}')



if __name__ == '__main__':
    main()
