import os
import os.path as osp
import numpy as np
import argparse
import readfile
from allocation import Allocation
from import_export import Import_export
from transportation import Greedy_price
from transportation import Greedy
from geopy.distance import great_circle

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

    #------------------------Import-export and Complaince--------------------------
    
    # import_export=readfile.read_import_export(example_path,'import_export.csv',crops)
    # for i in import_export.allocation.keys():
    #     for j in crops.keys():
    #         print(import_export.allocation[i][j])

    compliance=readfile.read_parameters(example_path,'parameters.csv')

    distances_numpy = calc_distances_numpy(district_numpy)

    allocation_numpy =  (1-compliance)*standard_allocation_numpy+compliance*new_allocation_numpy
 
    # # ############
    # # # for district_id in allocation.allocation.keys():
    # # #         for crop_id in allocation.allocation[district_id].keys():
    # # #             print(import_export.allocation[district_id][crop_id])
    # #             # pass
    # # ############
    # # Import_export
    # # import_export_obj = Import_export(allocation.allocation,import_export.allocation)
    # # import_export_obj.do_import_export()


    #--------------------------------Make_this_faster------------------------------------------
    stock_numpy=np.zeros((len(district_numpy),len(crops_numpy)))              # Districts->rows, crops->colunms 
    production_cost=np.zeros((len(district_numpy),len(crops_numpy)))          # Districts->rows, crops->colunms
    for i in range(len(crops_numpy)):
        stock_numpy[:,i] = allocation_numpy[:,i]*dist_crop_numpy[i][:,3]
        production_cost[:,i] = allocation_numpy[:,i]*dist_crop_numpy[i][:,2]
    
    # all_crop_arrival_prices = calc_arrival_prices(districts,crops)
    # transportation = Greedy_price(districts,crops,all_crop_arrival_prices)
    #-----------------------------------------GREEDY-------------------------------------------

    # checking feasibility
    flag,feasibility = solution_feasible(dist_crop_numpy, stock_numpy)
    if not flag:
        print('Solution not feasible : ',feasibility)
        return
    

    # print("")
    # print("Solution status : ",is_solution(dist_crop_numpy,stock_numpy))
    
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
    
    # print(stock_numpy)
    flag,feasibility = solution_feasible(dist_crop_numpy, stock_numpy)

    print("")
    print("Solution status : ",is_solution(dist_crop_numpy,stock_numpy))

    total_production_cost = np.sum(production_cost)
    total_revenue=0
    for i in range(dist_crop_numpy.shape[0]): # for all crops
        for j in range(stock_numpy.shape[0]):   #for all districts 
            total_revenue += max(stock_numpy[j,i],dist_crop_numpy[i,j,5])*dist_crop_numpy[i,j,4]*10


    print("")
    print("Total revenue from sale : ", round(total_revenue,2))
    print("")

    print("Total cost of production : ", round(total_production_cost,2))
    print("")

    print("Total transport cost : ", round(transportation_cost,2))
    print("")

    print("Total profit : ", round(total_revenue-total_production_cost-transportation_cost,2))
    print("")





if __name__ == '__main__':
    main()
