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
import googlemaps
from datetime import datetime
import requests 
def plot_results(name, avg_dict):
    barWidth = 0.02
    x_labels = [k/10 for k in range(11)]
    for state,i in zip(avg_dict.keys(),range(len(avg_dict.keys()))):
        #----------------Line graph---------------------
        # plt.plot(x_labels[2:],avg_dict[state][2:])
        plt.plot(x_labels,avg_dict[state])         
         
        #-----------------Bar graph----------------------
        # labels = []
        # for x in range(11):
        #     labels.append((x+i*0.2-0.4)/10)
        # plt.bar(labels, avg_dict[state], align='edge', width = barWidth)

    plt.title(name)
    plt.legend(list(avg_dict.keys()), shadow=True)
    plt.ylabel('In billion INR')
    plt.xlabel('Compliance ratio')

def parse_args():
    arg_parser = argparse.ArgumentParser()

    # input argument options
    arg_parser.add_argument(dest='example_path',
                            type=str,
                            help='Pass the path to the data folder')
    args = arg_parser.parse_args()
    return args

def calc_distances(no_dist,district_numpy):
    distances = np.zeros((no_dist,no_dist))
    for i in range(no_dist):
        for j in range(no_dist):
                if i == j:
                    # NEEDS TO BE UPDATED : Large value was taken to avoid transportation to self 
                    distances[i,j] = 10**5
                else: 
                    distances[i,j] = great_circle((district_numpy[i,2],district_numpy[i,1]),(district_numpy[j,2],district_numpy[j,1])).km
                    # print(distances[i,j])
    return distances


def calc_distances_wh(no_dist,district_numpy,no_warehouses, warehouses_numpy):
    distances = np.zeros((no_dist,no_warehouses))
    for i in range(no_dist):
        for j in range(no_warehouses):
                    distances[i,j] = great_circle((district_numpy[i,2],district_numpy[i,1]),(warehouses_numpy[j,4],warehouses_numpy[j,3])).km
                    # print(distances[i,j])
    return distances

def cal_road(api_key, origin, destination):
  

# Replace 'YOUR_API_KEY' with your actual Google Maps API key


# Replace these addresses with the addresses you want to calculate the distance between
# Construct the API request URL
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": destination,
        "mode": "driving",
        "units": "metric",
        "key": api_key
    }

# Make the API request
    response = requests.get(base_url, params=params)
    result = response.json()

# Extract the distance in meters (you can also get duration, etc.)
    distance_in_meters = result['rows'][0]['elements'][0]['distance']['value']

# Convert distance to kilometers
    distance_in_kilometers = distance_in_meters / 1000.0

    print(f"Distance between the two addresses: {distance_in_kilometers:.2f} km")
    return distance_in_kilometers


def calculate_road_distance(api_key, origin, destination):
    gmaps = googlemaps.Client(key=api_key)

    # Request distance matrix
    distance_matrix = gmaps.distance_matrix(origin, destination, mode="driving")

    # Check if the response is valid
    # my_dist = gmaps.distance_matrix(origin,destination)['rows'][0]['elements'][0]
    value=0
    if distance_matrix['status'] == 'OK':
        # Extract the distance in meters
        distance_in_meters = distance_matrix['rows'][0]['elements'][0]['distance']['value']
        
        # Convert meters to kilometers
        distance_in_kilometers = distance_in_meters / 1000
        value=distance_in_kilometers

        print(f"Road Distance between {origin} and {destination}: {distance_in_kilometers:.2f} km")
    else:
        print(f"Error: {distance_matrix['status']}")

    # value=my_dist/1000
    # print(f"Road Distance between {origin} and {destination}: {value:.2f} km")
    return value    
def calc_dist(no_dist, district_name): 
    distances = np.zeros((no_dist, no_dist))
    api_key = 'AIzaSyDPV0Aze7-htGc4IiYPryFeIHncBzTNwqo'
    for i in range(no_dist):
        for j in range(no_dist):
            origin = district_name[i,1]
            destination = district_name[j,1]
            if distances[j,i]!=0: 
                distances[i,j]=distances[j,i]
            elif i==j:
                distances[i,j]=10**5
            else: 
                distances[i,j]= calculate_road_distance(api_key, origin, destination)

    return distances            

def calc_distwh(no_dist, district_name, warehouse_address, no_warehouse): 
    distances = np.zeros((no_dist, no_warehouse))
    api_key = 'AIzaSyDPV0Aze7-htGc4IiYPryFeIHncBzTNwqo'
    for i in range(no_dist):
        for j in range(no_warehouse):
            origin = district_name[i,1]
            destination = warehouse_address[j,1]
            distances[i,j]= calculate_road_distance(api_key, origin, destination)

    return distances      

def read_files(example_path):
     
    #--------------------Districts------------------------------
    districts=readfile.read_districts(example_path,'district.csv')
    district_numpy = []
    district_name=[]
    for i in districts.keys():
       district_name.append([int(districts[i].id)-1, districts[i].name])
       district_numpy.append([int(districts[i].id)-1,districts[i].coordinates.longitude,districts[i].coordinates.latitude,districts[i].population])
    district_numpy = np.array(district_numpy)
    district_name = np.array(district_name)
    district_numpy = district_numpy.astype('float32') 
    print(district_numpy.shape)
    # print(district_numpy)
    print(district_name)

    #-------------------Warehouses-----------------------------
    warehouses=readfile.read_warehouse(example_path,'warehouse.csv')
    warehouses_numpy = []
    warehouse_address=[]
    print(warehouses.keys())
    for i in warehouses.keys():
       
       warehouse_address.append([int(warehouses[i].id)-1, warehouses[i].address, warehouses[i].name])
       warehouses_numpy.append([int(warehouses[i].id)-1,warehouses[i].capacity1,warehouses[i].cost,warehouses[i].coordinates.longitude,warehouses[i].coordinates.latitude])
    warehouses_numpy = np.array(warehouses_numpy)
    warehouse_address = np.array(warehouse_address)
    warehouses_numpy = warehouses_numpy.astype('float32') 
    print(warehouses_numpy.shape)

    #-----------------warehouses district wise---------------- 
    warehouses=readfile.read_ware_dist(example_path,'Wh_Allo_dist_f.csv')
    wh_dist_numpy = []
    # warehouse_address=[]
    print(warehouses.keys())
    for i in warehouses.keys():
       
    #    warehouse_address.append([int(warehouses[i].District_id)-1, warehouses[i].address, warehouses[i].name])
       wh_dist_numpy.append([warehouses[i].Bajra,warehouses[i].Cotton,warehouses[i].Groundnut, warehouses[i].Jowar, warehouses[i].Maiz, warehouses[i].Moong, warehouses[i].Onion, warehouses[i].Ragi, warehouses[i].Rice, warehouses[i].Soyabean, warehouses[i].Tur, warehouses[i].Wheat])
    #    warehouses_numpy.append([int(warehouses[i].District_id)-1,warehouses[i].Capacity,warehouses[i].Bajra,warehouses[i].Cotton,warehouses[i].Groundnut, warehouses[i].Jowar, warehouses[i].Maiz, warehouses[i].Moong, warehouses[i].Onion, warehouses[i].Ragi, warehouses[i].Rice, warehouses[i].Soyabean, warehouses[i].Tur, warehouses[i].Wheat])
    wh_dist_numpy = np.array(wh_dist_numpy)
    # warehouse_address = np.array(warehouse_address)
    wh_dist_numpy = wh_dist_numpy.astype('float32') 
    print(wh_dist_numpy.shape)

    #----------------------Crops---------------------------------
    crops = readfile.read_crops(example_path,'crop.csv')
    print(crops) 
    crops_numpy = []
    for i,j in zip(crops.keys(),range(len(crops))):
        crops_numpy.append([j,crops[i].transport_cost])  
    crops_numpy = np.array(crops_numpy)             #Rows->crops, columns-> transportation_costs
    # print(crops_numpy)
    print("----------------------------------------------------------------------------------------------")
    dist_crop_numpy = [] #rows->districts, columns->dist_no,crop_no,crop_id,cost,yeild,price,max_demand,min_demand, price_n, yield_n
    for district,j in zip(districts.values(),range(len(districts))):
        dist_crop_numpy.append(district.get_crop_details(crops,j))
    dist_crop_numpy = np.array(dist_crop_numpy)
    print(dist_crop_numpy.shape)
    print("Prashant")
    # print(dist_crop_numpy)
    #-------------------new_allocation---------------------------
    new_allocation = readfile.read_allocation(example_path,'new_allocation.csv',crops)
    new_allocation_numpy = []               #rows -> districts, columns -> crops
    for i in new_allocation.allocation.keys():
        temp = []
        for j in crops.keys():
            temp.append(new_allocation.allocation[i][j])  
        new_allocation_numpy.append(temp)
    new_allocation_numpy = np.array(new_allocation_numpy)
    print(new_allocation_numpy.shape)
    # print(new_allocation_numpy)

    #------------------standard_allocation-----------------------
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)
    standard_allocation_numpy = []           #rows -> districts, columns -> crops
    for i in standard_allocation.allocation.keys():
        temp = []
        for j in crops.keys():
            temp.append(standard_allocation.allocation[i][j])
        standard_allocation_numpy.append(temp)
    standard_allocation_numpy = np.array(standard_allocation_numpy)

    return district_numpy,wh_dist_numpy ,district_name, crops_numpy,dist_crop_numpy,standard_allocation_numpy,new_allocation_numpy,districts,crops,new_allocation, warehouse_address, warehouses_numpy

def allocation_to_csv(allocation):
    data = []
    index = [(i+1) for i in range(len(allocation.allocation.keys()))]
    # print(index)
    for i in allocation.allocation.keys():
        data.append(allocation.allocation[i])
    # print(f'len data:{data}')
    df = pd.DataFrame(data, index)
    df.to_csv('allocation_1.csv')

def print_to_file(districts,crops,alloc,new_allocation):
    for district,i in zip(districts.values(),range(len(districts.values()))):
                for crps,j in zip(list(crops.keys()),range(len(crops.keys()))):
                    new_allocation.allocation[district.id][crps]=alloc[i,j]
    # print(f"Allocation: {new_allocation.allocation}")
    allocation_to_csv(new_allocation)

def plot_all_metrics(alloc1,current_allocation_numpy,no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,no_warehouse,warehouse_capacity,warehouse_storage_cost,dist_warehouse,crop_yield_n,price_n,wh_id,x,alloc_1):
    recommended_allocation = alloc1
    recommended_allocation_1 = alloc_1
    d={'Revenue':[],'Production Cost':[],'Transportation Cost':[],'Profit':[]}
    for c in range(11):
        print(f'---------------------{c/10}-----------------------')
        # ---------------------------------Complince from distribution---------------------------------------
        # a = c/10-0.05
        # b = c/10+0.05
        # if a<0:
        #     a = 0
        # if b>1:
        #     b=1
        # p_sum = 0
        # r_sum = 0
        # t_sum = 0
        # Profit_sum = 0
        # no_iter = 50
        # for p in range(no_iter):
        #     compliance_matrix = (b - a) * np.random.random_sample(current_allocation_numpy.shape) + a
        #     ones_matrix = np.ones(current_allocation_numpy.shape)
        #     allocation =  np.multiply((ones_matrix-compliance_matrix),current_allocation_numpy)+np.multiply(compliance_matrix,recommended_allocation)
        #     p1,r1,t1,Profit1,alloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,allocation,optimize=0)
        #     p_sum += p1
        #     r_sum += r1
        #     t_sum += t1
        #     Profit_sum += Profit1
        # p = p_sum/no_iter
        # r = r_sum/no_iter
        # t = t_sum/no_iter
        # Profit = Profit_sum/no_iter
        # -------------------------------------For constant complinace----------------------------------------
        allocation = (1-c/10)*current_allocation_numpy + (c/10)*recommended_allocation
        allocation_1 = (1-c/10)*current_allocation_numpy + (c/10)*allocation 
        p,r,t,Profit,alloc,whalloc,alloc_2 = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,no_warehouse,warehouse_capacity,warehouse_storage_cost,dist_warehouse,crop_yield_n,price_n,wh_id,x,allocation,allocation_1,optimize=0)
        d['Revenue'].append(r/1e9)
        d['Production Cost'].append(p/1e9)
        d['Transportation Cost'].append(t/1e9) 
        d['Profit'].append(Profit/1e9)
        print(r/1e9,Profit/1e9) 
        # print(whalloc)
    plot_results ("Revenue, Costs and Profit vs Compliance ratio",d)
    plt.show()

def plot_max_min(no_dist,no_crop,cost,crop_yield,price,tc,distances,max_alloc,standard_allocation_numpy,ld,ud,no_warehouse,warehouse_capacity,warehouse_storage_cost,dist_warehouse,crop_yield_n,price_n,wh_id):
    for i in range(5):
        # ld_factor = (0.9-0.1*i)
        ud_factor = (1.1+0.1*i)
        # ud_factor = 1.2
        ld_factor=0.6
        ld_temp = ld
        ud_temp = ud
        ld_temp = ld*ld_factor/0.8 
        # ld_temp[:,1] = ld[:,1]*ld_factor/0.7
        ud_temp = ud*ud_factor/1.2
        # ud_temp[:,1] = ud[:,1]*ud_factor/1.2
        p,r,t,Profit,alloc,whalloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld_temp,ud_temp,tc,distances,max_alloc,no_warehouse,warehouse_capacity,warehouse_storage_cost,dist_warehouse,crop_yield_n,price_n,wh_id,allocation=0,optimize=1)
        profit_vector = profit_complaince(alloc,standard_allocation_numpy,no_dist,no_crop,cost,crop_yield,price,ld_temp,ud_temp,tc,distances,max_alloc,no_warehouse,warehouse_capacity,warehouse_storage_cost,dist_warehouse,crop_yield_n,price_n,wh_id)
        plt.plot(profit_vector,label=str(round(ld_factor*100))+'%, '+str(round(ud_factor*100))+'%')
    plt.xticks(list(range(11)),map(lambda x: x/10,list(range(11))))
    plt.ylabel('Profit in billion INR')
    plt.xlabel('Compliance ratio')
    plt.legend()
    plt.show()

#Calculates profit for different compliance values
def profit_complaince(alloc1,standard_allocation_numpy,no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,no_warehouse,warehouse_capacity,warehouse_storage_cost,dist_warehouse,crop_yield_n,price_n,wh_id):
    recommended_allocation = alloc1
    profit_vector = []
    for c in range(11):
        print(f'---------------------{c/10}-----------------------')
        allocation = (1-c/10)*standard_allocation_numpy + (c/10)*recommended_allocation
        p,r,t,Profit,alloc,whalloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,no_warehouse,warehouse_capacity,warehouse_storage_cost,dist_warehouse,crop_yield_n,price_n,wh_id,allocation,optimize=0)
        profit_vector.append(Profit/1e9)
    return profit_vector

def gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,LD,UD,TC,distances,max_alloc, no_warehouses, UBWarehouse, warehouse_storage_cost, distanceswh,crop_yield_n,price_n,wh_id,x_st,allocation,allocation_1,optimize):
    #----------------------Gurobi code----------------------
    model = Model(name='CROP')
    if optimize == 1:
        allocation = model.addVars(no_dist,no_crop,vtype=GRB.CONTINUOUS,lb = 0, name = 'allocation')  #For allocation 2018-19
        allocation_1 = model.addVars(no_dist,no_crop,vtype=GRB.CONTINUOUS,lb = 0, name = 'allocation_1')   #For allocation 2019-20
    x = model.addVars(no_dist,no_crop*no_dist,vtype=GRB.CONTINUOUS,lb = 0, name = 'x')  #For transportation for 2018-19
    x_1 = model.addVars(no_dist,no_crop*no_dist,vtype=GRB.CONTINUOUS,lb = 0, name = 'x_1')  #For transportation for 2019-20

    whalloacation = model.addVars(no_warehouses, no_crop,vtype=GRB.CONTINUOUS,lb = 0, name = 'whalloacation') # For Warehouse Allocation for 2018-19
    whalloacation_1 = model.addVars(no_warehouses, no_crop,vtype=GRB.CONTINUOUS,lb = 0, name = 'whalloacation_1') # For Warehouse Allocation for 2019-20
    wx = model.addVars(no_dist,no_crop*no_warehouses,vtype=GRB.CONTINUOUS,lb = 0, name = 'wx') # for WArehouse transportation 2018-19
    wx_1 = model.addVars(no_dist,no_crop*no_warehouses,vtype=GRB.CONTINUOUS,lb = 0, name = 'wx_1') # for WArehouse transportation 2018-19
    condition_1 = model.addVar(vtype=GRB.BINARY, name="condition_1")
    condition_2 = model.addVar(vtype=GRB.BINARY, name="condition_2")
    penalty=1e10
    #----------------------------------Adding constarints to gurobi model------------------------------------------------
    #Split these constraint equations LD and UD contsraint for 2018-19
    model.addConstrs((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i]\
                    for k in range(no_dist)))>=LD[i,j] for i in range(no_dist) for j in range(no_crop))
    
    # model.addConstrs(UD[i,j] - (allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i]\
    #                 for k in range(no_dist))) <= condition_1*penalty for i in range(no_dist) for j in range(no_crop))
    
    #Split these constraint equations LD and UD contsraint for 2019-20  #Update UD and LD for this years
    model.addConstrs((allocation_1[i,j]*crop_yield_n[i,j]-sum(x_1[i,j*no_dist+k] for k in range(no_dist))+sum(x_1[k,j*no_dist+i]\
                    for k in range(no_dist)))>=LD[i,j] for i in range(no_dist) for j in range(no_crop)) 
    
    # model.addConstrs(UD[i,j] - (allocation_1[i,j]*crop_yield_n[i,j]-sum(x_1[i,j*no_dist+k] for k in range(no_dist))+sum(x_1[k,j*no_dist+i]\
    #                 for k in range(no_dist)))<= condition_2*penalty   for i in range(no_dist) for j in range(no_crop))
    
     #Calculating cost of production and revenue for the state   
    cop = quicksum(allocation[i,j]*cost[i,j] for i in range(no_dist) for j in range(no_crop))  #2018-19
    cop_1 = quicksum(allocation_1[i,j]*cost[i,j] for i in range(no_dist) for j in range(no_crop))  #2019-20
    # Revenue = quicksum((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i] for k in range(no_dist)))*price[i,j] for i in range(no_dist) for j in range(no_crop))
    #revenuew of crop generated from 2018-19 stored sold in 2019-20
    Revenue_wh_1 = quicksum(whalloacation[i,j]*0.95*price_n[wh_id[i]-1,j] for i in range(no_warehouses) for j in range(no_crop)) 

    #revenuew of crop generated from 2018-19 stored sold in 2019-20
    Revenue_wh_2 = quicksum(whalloacation_1[i,j]*0.95*price[wh_id[i]-1,j] for i in range(no_warehouses) for j in range(no_crop))

    #revenew of warehouse stored crop 2017-18 and sold in 2018-19
    Revenue_wh = quicksum(x_st[i,j]*0.95*price[i,j] for i in range(no_dist) for j in range(no_crop)) 

    #stock available in market revenuw for 2018-19 
    Revenue_with_whs = quicksum((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))-sum(wx[i,j*no_warehouses+l] for l in range(no_warehouses))\
    +sum(x[k,j*no_dist+i] for k in range(no_dist)))*price[i,j] - condition_1*penalty for i in range(no_dist) for j in range(no_crop))
    #stock available in market revenew for 2019-20
    Revenue_with_whs_1 = quicksum((allocation_1[i,j]*crop_yield_n[i,j]-sum(x_1[i,j*no_dist+k] for k in range(no_dist))-sum(wx_1[i,j*no_warehouses+l] for l in range(no_warehouses))\
    +sum(x_1[k,j*no_dist+i] for k in range(no_dist)))*price_n[i,j] -condition_2*penalty for i in range(no_dist) for j in range(no_crop))

    Revenue_with_warehse=Revenue_with_whs+Revenue_wh
    rnw_war= Revenue_with_whs+Revenue_wh_2 + Revenue_wh_1+Revenue_with_whs_1+Revenue_wh # 2 year revenue 
    #Transporting crop j from district i -> k for 2018-19 
    Transport_cost = quicksum((TC[j]*distances[i,k]*x[i,j*no_dist+k]) for k in range(no_dist) for i in range(no_dist) for j in range(no_crop))
    
    # Warehouse transporation cost form i district to k warehouse 
    Transport_cost_wh = quicksum((TC[j]*distanceswh[i,k]*wx[i,j*no_warehouses+k]) for k in range(no_warehouses) for i in range(no_dist) for j in range(no_crop))
    

    # Transcportaion cost for 2019-20 
    Transport_cost_1 = quicksum((TC[j]*distances[i,k]*x_1[i,j*no_dist+k]) for k in range(no_dist) for i in range(no_dist) for j in range(no_crop))
    
    # Warehouse transporation cost form i district to k warehouse 
    Transport_cost_wh_1 = quicksum((TC[j]*distanceswh[i,k]*wx_1[i,j*no_warehouses+k]) for k in range(no_warehouses) for i in range(no_dist) for j in range(no_crop))
    

    
    totat_TC = Transport_cost + Transport_cost_1 + Transport_cost_wh + Transport_cost_wh_1 # 2 year transportation cost 

    # objective function with warehoouse 
    print("..........transporation cost of warehouse....................")
    # print(Transport_cost)
    print("\n")
    obj_fn_1 = rnw_war - totat_TC -cop - cop - warehouse_storage_cost -warehouse_storage_cost
    # obj_fn_wh = Revenue_with_warehse - cop - Transport_cost - Transport_cost_wh -warehouse_storage_cost 
    model.setObjective(obj_fn_1,GRB.MAXIMIZE)
    model.optimize() 

    #--------------------------------For printing the variable values---------------------------------------
    p = 0
    p_1 =0
    r = 0
    r_1 =0
    r_wh=0
    r_wh_1=0
    rr= Revenue_wh
    t = 0
    t_1 = 0
    t1=0
    t1_1 = 0
    Profit = 0
    Profit_1 =0 
    alloc=allocation
    alloc_1=allocation
    warealloc = [] 
    warealloc_1=[]
    if model.status == GRB.Status.OPTIMAL:
        values = model.getAttr("X", model.getVars()) #retrieive the valuse of decison variables retrive specified by "X"
        values = np.array(values) 
        if optimize == 1:
            # extracting first of aadvariables that is allocation for 2018-19
            alloc = values[0:no_dist*no_crop].reshape((no_dist,no_crop)) 

             # extracting first of aadvariables that is allocation  for 2019-20
            alloc_1 = values[no_dist*no_crop:no_dist*no_crop+no_dist*no_crop].reshape((no_dist,no_crop))

            #transported qty for 2018-19 
            transport_qty = values[no_dist*no_crop+no_dist*no_crop:no_dist*no_crop+no_crop*no_dist+no_dist*no_dist*no_crop].reshape((no_dist,no_dist*no_crop)) 
            # transportation quantity for 2019-20
            transport_qty_1 = values[no_dist*no_crop+no_crop*no_dist+no_dist*no_dist*no_crop: no_dist*no_crop+no_crop*no_dist+no_dist*no_dist*no_crop+no_dist*no_dist*no_crop].reshape((no_dist,no_dist*no_crop)) # transportation quantity
        
            # warehouse allocation for 2018-19
            warealloc = values[no_dist*no_crop+no_crop*no_dist+no_dist*no_dist*no_crop+no_dist*no_dist*no_crop:no_dist*no_crop+no_crop*no_dist+no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop].reshape((no_warehouses,no_crop)) 

            #warehouse allocation for 2019-20
            warealloc_1 = values[no_dist*no_crop+no_crop*no_dist+no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop:no_dist*no_crop+no_crop*no_dist+no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop+no_warehouses*no_crop].reshape((no_warehouses,no_crop)) 

            # warehouse transport for 2018-19
            ware_transport = values[no_dist*no_crop+no_crop*no_dist+no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop+no_warehouses*no_crop:no_dist*no_crop+no_crop*no_dist+no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop+no_warehouses*no_crop+no_dist*no_crop*no_warehouses].reshape((no_dist, no_crop*no_warehouses)) 
            
            #warehouse transport for 2019-20 
            ware_transport_1 = values[no_dist*no_crop+no_crop*no_dist+no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop+no_warehouses*no_crop+no_dist*no_crop*no_warehouses:no_dist*no_crop+no_crop*no_dist+no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop+no_warehouses*no_crop+no_dist*no_crop*no_warehouses+no_dist*no_crop*no_warehouses].reshape((no_dist, no_crop*no_warehouses)) 

            # print(f'Allocation: \n {alloc}')

            # print(f'Transported quantity: \n {transport_qty}')
            # np.savetxt('transport.csv', transport_qty, delimiter=',')
            # stock = np.zeros(alloc.shape)               #Final Stock
            # for i in range(no_dist):
            #     for j in range(no_crop):
            #         stock[i,j] = alloc[i,j]*crop_yield[i,j]-sum(transport_qty[i,j*no_dist+k] for k in range(no_dist))+sum(transport_qty[k,j*no_dist+i] for k in range(no_dist))
            # print(f'Stock: \n {stock}')
        else:
            # alloc_1 = values[0:no_dist*no_crop].reshape((no_dist,no_crop))

            #transported qty for 2018-19 
            transport_qty = values[0:no_dist*no_dist*no_crop].reshape((no_dist,no_dist*no_crop)) 
            # transportation quantity for 2019-20
            transport_qty_1 = values[no_dist*no_dist*no_crop:no_dist*no_dist*no_crop+no_dist*no_dist*no_crop].reshape((no_dist,no_dist*no_crop)) # transportation quantity
        
            # warehouse allocation for 2018-19
            warealloc = values[no_dist*no_dist*no_crop+no_dist*no_dist*no_crop:no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop].reshape((no_warehouses,no_crop)) 

            #warehouse allocation for 2019-20
            warealloc_1 = values[no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop:no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop+no_warehouses*no_crop].reshape((no_warehouses,no_crop)) 

            # warehouse transport for 2018-19
            ware_transport = values[no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop+no_warehouses*no_crop:no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop+no_warehouses*no_crop+no_dist*no_crop*no_warehouses].reshape((no_dist, no_crop*no_warehouses)) 
            
            #warehouse transport for 2019-20 
            ware_transport_1 = values[no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop+no_warehouses*no_crop+no_dist*no_crop*no_warehouses:no_dist*no_dist*no_crop+no_dist*no_dist*no_crop+no_warehouses*no_crop+no_warehouses*no_crop+no_dist*no_crop*no_warehouses+no_dist*no_crop*no_warehouses].reshape((no_dist, no_crop*no_warehouses)) 
    
        #coc 2018-19
        p = sum(alloc[i,j]*cost[i,j] for i in range(no_dist) for j in range(no_crop))
        #coc 2019-20
        p_1 = sum(alloc_1[i,j]*cost[i,j] for i in range(no_dist) for j in range(no_crop))  
        #revenue 2018-19
        r = sum((alloc[i,j]*crop_yield[i,j]-sum(transport_qty[i,j*no_dist+k] for k in range(no_dist))+sum(transport_qty[k,j*no_dist+i] for k in range(no_dist))-sum(ware_transport[i,j*no_warehouses+k] for k in range(no_warehouses)))*price[i,j] for i in range(no_dist) for j in range(no_crop))
       #revenue 2019-20
        r_1 = sum((alloc_1[i,j]*crop_yield_n[i,j]-sum(transport_qty_1[i,j*no_dist+k] for k in range(no_dist))+sum(transport_qty_1[k,j*no_dist+i] for k in range(no_dist))-sum(ware_transport_1[i,j*no_warehouses+k] for k in range(no_warehouses)))*price_n[i,j] for i in range(no_dist) for j in range(no_crop))
     
        #revenue by warehouse 2018-19 
        r_wh = sum(warealloc[i,j]*0.95*price_n[wh_id[i]-1,j] for i in range(no_warehouses) for j in range(no_crop))

        #revenue by warehouse 2019-20
        r_wh_1 = sum(warealloc_1[i,j]*0.95*price_n[wh_id[i]-1,j] for i in range(no_warehouses) for j in range(no_crop))
        #transportation 2018-19      
        t = sum((TC[j]*distances[i,k]*transport_qty[i,j*no_dist+k]) for k in range(no_dist) for i in range(no_dist) for j in range(no_crop))
       
       #transportation 2018-19      
        t_1 = sum((TC[j]*distances[i,k]*transport_qty_1[i,j*no_dist+k]) for k in range(no_dist) for i in range(no_dist) for j in range(no_crop))
      
        # warehouse transportation 2018-19
        t1 = sum((TC[j]*distanceswh[i,k]*ware_transport[i,j*no_warehouses+k]) for k in range(no_warehouses) for i in range(no_dist) for j in range(no_crop))
        
        # warehouse transportation 2019-20
        t1_1 = sum((TC[j]*distanceswh[i,k]*ware_transport_1[i,j*no_warehouses+k]) for k in range(no_warehouses) for i in range(no_dist) for j in range(no_crop))
       
        # Profit = r+r_1+r_wh+r_wh_1-p-p_1-t-t_1 -t1- t1_1
        Profit = r + r_wh -p - t - t1
        Profit_1 = r_1 + r_wh_1 -t_1 -t1_1-p_1
        # print(Profit_1)
        # print(r_1+r_wh_1)
    if Profit == 0.0:
        # model.computeIIS()
        print(Profit)
    return p, r+r_wh , t, Profit,alloc,warealloc,alloc_1    
    # return p+p_1,r+r_1+r_wh+r_wh_1,t+t_1+t1+t1_1,Profit+Profit_1,alloc,warealloc,alloc_1

#---------------------------------For debugging compliance variablility problem----------------------
def gurobi_optimizer_copy(no_dist,no_crop,cost,crop_yield,price,LD,UD,TC,distances,max_alloc,allocation,optimize):
    #----------------------Gurobi code----------------------
    model = Model(name='CROP')
    if optimize == 1:
        allocation = model.addVars(no_dist,no_crop,vtype=GRB.CONTINUOUS,lb = 0, name = 'allocation')  #For allocation
    x = model.addVars(no_dist,no_crop*no_dist,vtype=GRB.CONTINUOUS,lb = 0, name = 'x')  #For transportation 
    
    #----------------------------------Adding constarints to gurobi model------------------------------------------------
    #Split these constraint equations 
    model.addConstrs((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i]\
                    for k in range(no_dist)))>=LD[i,j] for i in range(no_dist) for j in range(no_crop))
    
    model.addConstrs((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i]\
                    for k in range(no_dist)))<=UD[i,j] for i in range(no_dist) for j in range(0,no_crop,1))

    if optimize == 1: 
        model.addConstrs((sum(allocation[i,j] for j in range(no_crop))<=max_alloc[i]) for i in range(no_dist))

    #Calculating cost of production and revenue for the state
    cop = quicksum(allocation[i,j]*cost[i,j] for i in range(no_dist) for j in range(no_crop))
    Revenue = quicksum((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i] for k in range(no_dist)))*price[i,j] for i in range(no_dist) for j in range(no_crop))
    
    #Transporting crop j from district i -> k
    Transport_cost = quicksum((TC[j]*distances[i,k]*x[i,j*no_dist+k]) for k in range(no_dist) for i in range(no_dist) for j in range(no_crop))
    obj_fn =  Revenue - cop -  Transport_cost
    model.setObjective(obj_fn,GRB.MAXIMIZE)
    model.optimize()
    
    #--------------------------------For printing the variable values---------------------------------------
    p = 0
    r = 0
    t = 0
    Profit = 0
    alloc=allocation 
    if model.status == GRB.Status.OPTIMAL:
        values = model.getAttr("X", model.getVars())
        values = np.array(values)
        if optimize == 1:
            alloc = values[0:no_dist*no_crop].reshape((no_dist,no_crop))
            transport_qty = values[no_dist*no_crop:].reshape((no_dist,no_dist*no_crop))
        else:
            transport_qty = values.reshape((no_dist,no_dist*no_crop))
        p = sum(alloc[i,j]*cost[i,j] for i in range(no_dist) for j in range(no_crop))
        r = sum((alloc[i,j]*crop_yield[i,j]-sum(transport_qty[i,j*no_dist+k] for k in range(no_dist))+sum(transport_qty[k,j*no_dist+i] for k in range(no_dist)))*price[i,j] for i in range(no_dist) for j in range(no_crop))
        t = sum((TC[j]*distances[i,k]*transport_qty[i,j*no_dist+k]) for k in range(no_dist) for i in range(no_dist) for j in range(no_crop))
        Profit = r-p-t
    if Profit == 0.0:
        model.computeIIS()
    return p,r,t,Profit,alloc

#---------------------------------------Gurobi-----------------------------------------
def indices_of_top_five(arr):
    # Convert the input list to a NumPy array if it's not already one
    np_arr = np.array(arr)
    
    # Find the indices of the five largest elements
    # np.argsort returns the indices that would sort the array, so we take the last five for the largest values
    # [::-1] reverses the array to get the largest values first, then we take the first five
    indices = np.argsort(np_arr)[-5:][::-1]
    
    return indices
#....................................... top five expensive crop ...............................

def main():
    # get the start time
    st = time.time()
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'
    
    district_numpy,wh_dist_numpy,district_name, crops_numpy,dist_crop_numpy,standard_allocation_numpy,new_allocation_numpy,districts,crops,new_allocation, warehouse_address, warehouses_numpy = read_files(example_path)
    
    # For now area under cultivation is taken as the past years area
    no_dist= district_numpy.shape[0]
    #----------------warehouse calculation-------------------------------------------------------------
    no_warehouse=warehouses_numpy.shape[0]
    warehouse_storage_cost=0
    warehouse_capacity=np.zeros(no_warehouse)
    wh_id=np.array([17,15,14,14,3,28,5,28,6,15,5,26,25,13,14,21,13,3,15,12,23,15,15,25,12,11,15,13,19,6,21,22,23,15,26,15,20,4,28,3,12,23,30,12,21,25,15,16,26,23,12,17,15,4,15,11,23,11,18,30,20])
    # wh_id = np.array(wh_id1)
    print(no_warehouse)
    print(warehouses_numpy.shape)

    for i in range(no_warehouse):
        warehouse_capacity[i]=warehouses_numpy[i][1]*0.5
        # wh_id[i]=warehouses_numpy[i][0]
        # print(wh_id[i])
        warehouse_storage_cost=warehouse_storage_cost+warehouses_numpy[i][2]

    dist_warehouse = calc_distances_wh(no_dist,district_numpy, no_warehouse,warehouses_numpy)

    #----------------------------------------------------------------------------------------------------------------    

    new_alloc = standard_allocation_numpy   
    # Finding total cultivation area which would be maximum croped area districtwise 
    max_alloc = np.sum(new_alloc,axis=1)
    print(max_alloc.shape)
  
    no_crop = crops_numpy.shape[0]

    distances = calc_distances(no_dist,district_numpy)
    # dist_pv = calc_dist(no_dist,district_name)
    print(warehouse_storage_cost)
#    print(distances)
    print("..................................")
    # print(dist_pv)

    # for i in range(no_dist):
    #     for j in range(no_dist):
    #         print(distances[i,j], dist_pv[i,j])


    quintal_to_ton = 10
    cost = dist_crop_numpy[:,:,2]               #Cost per crop per ton
    crop_yield = dist_crop_numpy[:,:,3]         #Yeild tonnes per hectare
    crop_yield_n = dist_crop_numpy[:,:,7] 
    price = dist_crop_numpy[:,:,4]*quintal_to_ton        #price in tonnes
    price_n = dist_crop_numpy[:,:,8]*quintal_to_ton
    ld = dist_crop_numpy[:,:,6]*(0.8/0.7)*0.6       #Minimum demand of district in tonnes (Originally 70% of district-wise production)
    ud = dist_crop_numpy[:,:,5]*(1.2/1.1)*1.2 #Maximum dwmand of district in tonnes (110% of district-wise production)
    tc = crops_numpy[:,1]     #Transportation cost in rupees per ton per km
    x= wh_dist_numpy*0.9
    y= ld*0.1                         #Transportation cost in rupees per ton per km
    print(ld.shape)
    print(warehouses_numpy.shape)
    print(price_n.shape)
    expensive_crop=[]
    for row in price: 
        indice = indices_of_top_five(row)
        print("Indices of the five maximum elements:", indice)
        expensive_crop.append(indice)
    print(expensive_crop[21][3])    
    
    ld_updated = ld - x +y
    ud_updated = ud -x + wh_dist_numpy
    #-------------------------------------Making demand proportional to population-----------------------------------
    # LD_factor = 0.9
    # UD_factor = 1.1
    # total_population = np.sum(district_numpy[:,3])
    # district_wise_population = district_numpy[:,3].reshape(no_dist,1)
    # total_produce = np.sum(np.multiply(crop_yield,new_alloc),axis=0).reshape(no_crop,1)
    # produce_per_unit_pop = total_produce/total_population
    # UD = np.matmul(district_wise_population,produce_per_unit_pop.T)*UD_factor
    # LD = (UD/UD_factor)*LD_factor

    #-----------------------------------------Experiments for ACM Compass-------------------------------------------
    # Changing price of one crop
    # price[:,1] *= 2
    # p->production cost, r->revenue, t-> transportation cost
    
    #-------------------------------------------Basic optimizer code-------------------------------------------------
    p,r,t,Profit,alloc,whalloc,alloc_1 = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld_updated,ud_updated,tc,distances,max_alloc,no_warehouse,warehouse_capacity,warehouse_storage_cost,dist_warehouse,crop_yield_n,price_n,wh_id,x,allocation=0,allocation_1=0,optimize=1)
    print(Profit)
    print(r)
    # print(alloc)
    # print(alloc_1)
    # print(warehouse_storage_cost)
    # print(f"COP = {p}")
    # print(f'Revenue = {r}')
    # print(f'Transport cost = {t}')
    # print(f'Profit = {Profit}') 
    # alloc_1yr = 0.1*alloc + 0.9*new_allocation_numpy
    # alloc_2yr = 0.1*alloc_1 + 0.9*new_allocation_numpy 
    # p_x,r_x,t_x,Profit_x,alloc_x,whalloc_x,alloc_1_x = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld_updated,ud_updated,tc,distances,max_alloc,no_warehouse,warehouse_capacity,warehouse_storage_cost,dist_warehouse,crop_yield_n,price_n,wh_id,x,alloc_1yr,alloc_2yr,optimize=0)
        
    # print(r_x)
    # print(Profit_x) 
    # print(alloc_x - ud_updated) 
    
    # d={'Revenue':[],'Production Cost':[],'Transportation Cost':[],'Profit':[]}
    # A = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # for i in range(11): 
    #     print("compliance")
    #     print(A[i])
    #     alloc_1yr = A[i]*alloc + (1-A[i])*new_allocation_numpy
    #     alloc_2yr = A[i]*alloc_1 + (1-A[i])*new_allocation_numpy
    #     p_x,r_x,t_x,Profit_x,alloc_x,whalloc_x,alloc_1_x = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld_updated,ud_updated,tc,distances,max_alloc,no_warehouse,warehouse_capacity,warehouse_storage_cost,dist_warehouse,crop_yield_n,price_n,wh_id,x,alloc_1yr,alloc_2yr,optimize=0)
        
    #     print(r_x)
    #     print(Profit_x)
    #     print(alloc_x -ud_updated<=0)
    #     print(alloc_1_x - ud_updated<=0)
    #     d['Revenue'].append(r_x/1e9)
    #     d['Production Cost'].append(p_x/1e9)
    #     d['Transportation Cost'].append(t_x/1e9) 
    #     d['Profit'].append(Profit_x/1e9)
    #     print(r_x/1e9,Profit_x/1e9) 

    # plot_results ("Revenue, Costs and Profit vs Compliance ratio",d)
    # plt.show()    
    #----------------------------------Transportation cost vary-----------------------------------------------------
    # x = []
    # profit_arr = []
    # tc_arr = []
    # # factors = [x for ]
    # for i in range(8):
    #     factor = (0.25+.25*i)
    #     x.append(factor)
    #     tc_temp = tc*factor
    #     p,r,t,Profit,alloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc_temp,distances,max_alloc,allocation=0,optimize=1)
    #     profit_arr.append(Profit/1e9)
    #     # tc_arr.append(t/1e9)
    # plt.plot(x,profit_arr,label='Profit')
    # # plt.plot(x,tc_arr,label='Transport cost')
    # plt.ylabel('Profit in billion INR')
    # plt.xlabel('Multiplication factor for transportation cost')
    # # plt.title('Effect of varying transport cost')
    # plt.show()


    #----------------------------------writing allocation to csv-----------------------------------------------------
    # print_to_file(districts,crops,alloc,new_allocation)
    
    #-----------------------------------Plotting revenue,TC,COP,Profit----------------------------------------
    plot_all_metrics(alloc,new_allocation_numpy,no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,no_warehouse,warehouse_capacity,warehouse_storage_cost,dist_warehouse,crop_yield,price_n,wh_id,x,alloc_1)
    
    # print(whalloc)
    # # -------------------------------------For constant complinace----------------------------------------
    # # allocation = (1-c/10)*current_allocation_numpy + (c/10)*recommended_allocation
    # p,r,t,Profit,alloc = gurobi_optimizer_copy(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,allocation,optimize=0)
    # d['Revenue'].append(r/1e9)
    # print(f'Revenue : {r/1e9}')
    # d['Production Cost'].append(p/1e9)
    # d['Transportation Cost'].append(t/1e9)
    # d['Profit'].append(Profit/1e9)
    # print(f'Profit: {Profit/1e9}')
    
    #----------------------------------Plotting profit for different max-min demands---------------------------------------
    # plot_max_min(no_dist,no_crop,cost,crop_yield,price,tc,distances,max_alloc,standard_allocation_numpy,ld,ud,no_warehouse,warehouse_capacity,warehouse_storage_cost,dist_warehouse,crop_yield,price_n,wh_id)

if __name__ == '__main__':
    main()