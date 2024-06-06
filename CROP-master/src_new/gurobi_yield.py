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
    return distances

def read_files(example_path):
     
    #--------------------Districts------------------------------
    districts=readfile.read_districts(example_path,'district.csv')
    district_numpy = []
    for i in districts.keys():
        district_numpy.append([int(districts[i].id)-1,districts[i].coordinates.longitude,districts[i].coordinates.latitude,districts[i].population])
    district_numpy = np.array(district_numpy)
    district_numpy = district_numpy.astype('float32') 
    # print(district_numpy.shape)
    print(district_numpy)
    
    #-------------------Warehouses-----------------------------
    warehouses=readfile.read_ware_dist(example_path,'Wh_Allo_dist_f.csv')
    warehouses_numpy = []
    dist_ware_cap=[]
    # warehouse_address=[]
    print(warehouses.keys())
    for i in warehouses.keys():
       
    #    warehouse_address.append([int(warehouses[i].District_id)-1, warehouses[i].address, warehouses[i].name])
       warehouses_numpy.append([warehouses[i].Bajra,warehouses[i].Cotton,warehouses[i].Groundnut, warehouses[i].Jowar, warehouses[i].Maiz, warehouses[i].Moong, warehouses[i].Onion, warehouses[i].Ragi, warehouses[i].Rice, warehouses[i].Soyabean, warehouses[i].Tur, warehouses[i].Wheat])
       dist_ware_cap.append(warehouses[i].Capacity1) 
    #    warehouses_numpy.append([int(warehouses[i].District_id)-1,warehouses[i].Capacity,warehouses[i].Bajra,warehouses[i].Cotton,warehouses[i].Groundnut, warehouses[i].Jowar, warehouses[i].Maiz, warehouses[i].Moong, warehouses[i].Onion, warehouses[i].Ragi, warehouses[i].Rice, warehouses[i].Soyabean, warehouses[i].Tur, warehouses[i].Wheat])
    warehouses_numpy = np.array(warehouses_numpy)
    dist_ware_cap_numpy = np.array(dist_ware_cap)
    # warehouse_address = np.array(warehouse_address)
    warehouses_numpy = warehouses_numpy.astype('float32') 
    dist_ware_cap_numpy= dist_ware_cap_numpy.astype('float32') 
    print(warehouses_numpy.shape)
    #----------------------Crops---------------------------------
    crops = readfile.read_crops(example_path,'crop.csv')
    # print(crops) 
    crops_numpy = []
    for i,j in zip(crops.keys(),range(len(crops))):
        crops_numpy.append([j,crops[i].transport_cost])  
    crops_numpy = np.array(crops_numpy)             #Rows->crops, columns-> transportation_costs
    # print(crops_numpy)
    print("----------------------------------------------------------------------------------------------")
    dist_crop_numpy = [] #rows->districts, columns->dist_no,crop_no,crop_id,cost,yeild,price,max_demand,min_demand
    for district,j in zip(districts.values(),range(len(districts))):
        dist_crop_numpy.append(district.get_crop_details(crops,j))
    dist_crop_numpy = np.array(dist_crop_numpy)
    print(dist_crop_numpy.shape)
    print(".....................stop ho ja..........................")
    # print(dist_crop_numpy.shape)
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
    print(new_allocation_numpy)

    #------------------standard_allocation-----------------------
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)
    standard_allocation_numpy = []           #rows -> districts, columns -> crops
    for i in standard_allocation.allocation.keys():
        temp = []
        for j in crops.keys():
            temp.append(standard_allocation.allocation[i][j])
        standard_allocation_numpy.append(temp)
    standard_allocation_numpy = np.array(standard_allocation_numpy)

    return district_numpy,warehouses_numpy,crops_numpy,dist_crop_numpy,standard_allocation_numpy,new_allocation_numpy,districts,crops,new_allocation,dist_ware_cap_numpy

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

def plot_all_metrics(alloc1,current_allocation_numpy,no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,stored_warehouse,dist_war_cap,price_n,y,expensive_crop):
    recommended_allocation = alloc1
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
        p,r,t,Profit,alloc,whalloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,stored_warehouse, dist_war_cap,price_n,y,expensive_crop, allocation,optimize=0)
        d['Revenue'].append(r/1e9)
        d['Production Cost'].append(p/1e9)
        d['Transportation Cost'].append(t/1e9)
        d['Profit'].append(Profit/1e9)
        print(r/1e9,Profit/1e9)
    plot_results ("Revenue, Costs and Profit vs Compliance ratio",d)
    plt.show()

def plot_max_min(no_dist,no_crop,cost,crop_yield,price,tc,distances,max_alloc,standard_allocation_numpy,ld,ud,stored_warehouse,warehouse_numpy,price_n,y,expensive_crop):
    for i in range(5):
        ld_factor = (0.9-0.1*i)
        # ud_factor = (1.1+0.1*i)
        ud_factor = 1.2
        ld_temp = ld
        ud_temp = ud
        # ld_temp[:,1] = ld[:,1]*ld_factor/0.8
        ld_temp = ld*ld_factor/0.8
        # ud_temp = ud*ud_factor/1.1
        p,r,t,Profit,alloc,whalloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld_temp,ud_temp,tc,distances,max_alloc,stored_warehouse,warehouse_numpy,price_n,y,expensive_crop, allocation=0,optimize=1)
        profit_vector = profit_complaince(alloc,standard_allocation_numpy,no_dist,no_crop,cost,crop_yield,price,ld_temp,ud_temp,tc,distances,max_alloc,stored_warehouse,warehouse_numpy,price_n,y,expensive_crop)
        plt.plot(profit_vector,label=str(round(ld_factor*100))+'%, '+str(round(ud_factor*100))+'%')
    plt.xticks(list(range(11)),map(lambda x: x/10,list(range(11))))
    plt.ylabel('Profit in billion INR')
    plt.xlabel('Compliance ratio')
    plt.legend()
    plt.show()

#Calculates profit for different compliance values
def profit_complaince(alloc1,standard_allocation_numpy,no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,stored_warehouse,warehouse_numpy,price_n,y,expensive_crop):
    recommended_allocation = alloc1
    profit_vector = []
    for c in range(11):
        print(f'---------------------{c/10}-----------------------')
        allocation = (1-c/10)*standard_allocation_numpy + (c/10)*recommended_allocation
        p,r,t,Profit,alloc,whalloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,stored_warehouse,warehouse_numpy,price_n,y,expensive_crop, allocation,optimize=0)
        profit_vector.append(Profit/1e9)
    return profit_vector

def gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,LD,UD,TC,distances,max_alloc,stored_warehouse,dist_war_cap,price_n,y,expensive_crop, allocation,optimize):
    #----------------------Gurobi code----------------------
    no_exp_crop=7
    model = Model(name='CROP')
    if optimize == 1:
        allocation = model.addVars(no_dist,no_crop,vtype=GRB.CONTINUOUS,lb = 0, name = 'allocation')  #For allocation
    x = model.addVars(no_dist,no_crop*no_dist,vtype=GRB.CONTINUOUS,lb = 0, name = 'x')  #For transportation
    whalloacation = model.addVars(no_dist, no_exp_crop,vtype=GRB.CONTINUOUS,lb = 0, name = 'whalloacation') # For Warehouse Allocation
    wx = model.addVars(no_dist,no_exp_crop*no_dist,vtype=GRB.CONTINUOUS,lb = 0, name = 'wx') # for WArehouse crop allocation  
    
    #----------------------------------Adding constarints to gurobi model------------------------------------------------
    #Split these constraint equations 
    model.addConstrs((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i]\
                    for k in range(no_dist)))>=LD[i,j]  for i in range(no_dist) for j in range(no_crop))
    
    model.addConstrs((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i]\
                    for k in range(no_dist)))<=UD[i,j] for i in range(no_dist) for j in range(no_crop))
    
    model.addConstrs((sum(whalloacation[i,j] for j in range(no_exp_crop))<=dist_war_cap[i]) for i in range(no_dist))
    # model.addConstrs((whalloacation[i,j] >= y[i,j]) for i in range(no_dist) for j in range(no_crop) )  

    if optimize == 1: 
        model.addConstrs((sum(allocation[i,j] for j in range(no_crop))<=max_alloc[i]) for i in range(no_dist))
    
    #Calculating cost of production and revenue for the state
    Revenue_wh = quicksum(whalloacation[i,j]*0.95*price_n[i,expensive_crop[j]] for i in range(no_dist) for j in range(no_exp_crop)) 
    Transport_cost_wh = quicksum((TC[expensive_crop[j]]*distances[i,k]*wx[i,j*no_dist+k]) for k in range(no_dist) for i in range(no_dist) for j in range(no_exp_crop))

    cop = quicksum(allocation[i,j]*cost[i,j] for i in range(no_dist) for j in range(no_crop))
    Revenue = quicksum((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i] for k in range(no_dist)))*price[i,j] for i in range(no_dist) for j in range(no_crop))
    
    #Transporting crop j from district i -> k
    Transport_cost = quicksum((TC[j]*distances[i,k]*x[i,j*no_dist+k]) for k in range(no_dist) for i in range(no_dist) for j in range(no_crop))
    obj_fn =  Revenue - cop -  Transport_cost
    obj_fn_wh = Revenue_wh + Revenue- cop - Transport_cost - Transport_cost_wh -81315651.0
    model.setObjective(obj_fn,GRB.MAXIMIZE)
    model.optimize()

    #--------------------------------For printing the variable values---------------------------------------
    p = 0
    r = 0
    t = 0
    Profit = 0
    alloc=allocation
    warealloc=[]
    if model.status == GRB.Status.OPTIMAL:
        values = model.getAttr("X", model.getVars()) #retrieive the valuse of decison variables retrive specified by "X"
        values = np.array(values) 
        if optimize == 1:
            alloc = values[0:no_dist*no_crop].reshape((no_dist,no_crop)) # extracting first of aadvariables that is allocation 
            transport_qty = values[no_dist*no_crop:no_crop*no_dist+no_dist*no_dist*no_crop].reshape((no_dist,no_dist*no_crop)) # transportation quantity 
            # warehouse allocation 
            warealloc = values[no_crop*no_dist+no_dist*no_dist*no_crop:no_crop*no_dist+no_dist*no_dist*no_crop+no_dist*no_exp_crop].reshape((no_dist,no_exp_crop)) 
            # warehouse transport
            ware_transport = values[no_crop*no_dist+no_dist*no_dist*no_crop+no_dist*no_exp_crop:no_crop*no_dist+no_dist*no_dist*no_crop+no_dist*no_exp_crop+no_dist*no_exp_crop*no_dist].reshape((no_dist, no_exp_crop*no_dist)) 
            # print(f'Allocation: \n {alloc}')
            # print(f'Allocation: \n {alloc}')
            # print(f'Transported quantity: \n {transport_qty}')
            # np.savetxt('transport.csv', transport_qty, delimiter=',')
            # stock = np.zeros(alloc.shape)               #Final Stock
            # for i in range(no_dist):
            #     for j in range(no_crop):
            #         stock[i,j] = alloc[i,j]*crop_yield[i,j]-sum(transport_qty[i,j*no_dist+k] for k in range(no_dist))+sum(transport_qty[k,j*no_dist+i] for k in range(no_dist))
            # print(f'Stock: \n {stock}')
        else:
            transport_qty = values[0:no_dist*no_dist*no_crop].reshape((no_dist,no_dist*no_crop))
            # warehouse allocation 
            warealloc = values[no_dist*no_dist*no_crop:no_dist*no_dist*no_crop+no_dist*no_exp_crop].reshape((no_dist,no_exp_crop)) 
            # warehouse transport
            ware_transport = values[no_dist*no_dist*no_crop+no_dist*no_exp_crop:no_dist*no_dist*no_crop+no_dist*no_exp_crop+no_dist*no_exp_crop*no_dist].reshape((no_dist, no_exp_crop*no_dist)) 
            # print(f'Allocation: \n {alloc}')
        p = sum(alloc[i,j]*cost[i,j] for i in range(no_dist) for j in range(no_crop))
        r = sum((alloc[i,j]*crop_yield[i,j]-sum(transport_qty[i,j*no_dist+k] for k in range(no_dist))+sum(transport_qty[k,j*no_dist+i] for k in range(no_dist)))*price[i,j] for i in range(no_dist) for j in range(no_crop))
        t = sum((TC[j]*distances[i,k]*transport_qty[i,j*no_dist+k]) for k in range(no_dist) for i in range(no_dist) for j in range(no_crop))
        Profit = r-p-t
    if Profit == 0.0:
        model.computeIIS()
    return p,r,t,Profit,alloc,warealloc

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
    indices = np.argsort(np_arr)[-7:][::-1]
    # indices = np.argsort(np_arr)[7:]
    
    return indices 

def array_to_csv(array, csv_file_path, column_names=None):
    """
    Converts a NumPy array to a CSV file.

    :param array: NumPy array to be converted
    :param csv_file_path: str, the path where the CSV file will be saved
    :param column_names: list of str, the column names for the CSV file (optional)
    """
    # Convert the NumPy array to a DataFrame
    df = pd.DataFrame(array, columns=column_names)

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)


def main():
    # get the start time
    st = time.time()
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'
    
    district_numpy,warehouses_numpy, crops_numpy,dist_crop_numpy,standard_allocation_numpy,new_allocation_numpy,districts,crops,new_allocation,dist_ware_cap_numpy = read_files(example_path)
    
    # For now area under cultivation is taken as the past years area
    new_alloc = standard_allocation_numpy   
    # Finding total cultivation area which would be maximum croped area districtwise 
    max_alloc = np.sum(new_alloc,axis=1)
    print(max_alloc.shape)
    no_dist= district_numpy.shape[0]
    no_crop = crops_numpy.shape[0]

    distances = calc_distances(no_dist,district_numpy)

    quintal_to_ton = 10
    cost = dist_crop_numpy[:,:,2]               #Cost per crop per ton
    crop_yield = dist_crop_numpy[:,:,3]         #Yeild tonnes per hectare
    price = dist_crop_numpy[:,:,4]*quintal_to_ton        #price in tonnes
    price_n = dist_crop_numpy[:,:,8]*quintal_to_ton
    ld = dist_crop_numpy[:,:,6]*(0.8/0.7)
    ld1 = dist_crop_numpy[:,:,7]          #Minimum demand of district in tonnes (Originally 70% of district-wise production)
    ud = dist_crop_numpy[:,:,5]*(1.2/1.1)             #Maximum dwmand of district in tonnes (110% of district-wise production)
    tc = crops_numpy[:,1]  
    x= warehouses_numpy*0.9
    y= ld*0.1                         #Transportation cost in rupees per ton per km
    print(crop_yield.shape)
    print(ld1.shape)
    print(warehouses_numpy.shape) 
    ld_updated = ld - x +y
    ud_updated = ud -x + warehouses_numpy
    expensive_crop=[]
    for row in price_n: 
        indice = indices_of_top_five(row)
        print("Indices of the five maximum elements:", indice)
        expensive_crop.append(indice)

    stored_warehouse = np.zeros((30, 12)) 
    percentage_array=np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    
    for i in range(30): 
        if dist_ware_cap_numpy[i]==0: 
            continue
        else: 
            for j in range(5): 
                k= expensive_crop[i][j]
                stored_warehouse[i][k]=percentage_array[j]*dist_ware_cap_numpy[i]
        # print(stored_warehouse[i])      
    #............... warehouse stored conversion to csv ..........................

    # stored_warehouse_numpy = np.array(stored_warehouse)
    # stored_warehouse_numpy= stored_warehouse_numpy.astype('float32')   
    # csv_file_path = '/home/prashant/CROP-master/examples/10_crops/stored.csv'

    # # Optionally, specify the column names
    # column_names = ['a', 'b','c','d','e','f','g','h','i','j','k','l']

    # # Convert the array to a CSV file
    # array_to_csv(stored_warehouse_numpy, csv_file_path, column_names)

    # print(f"Array has been saved to {csv_file_path}")
    #....................................end ....................................
    
    # ld_updated=ld_updated+stored_warehouse_numpy
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
    p,r,t,Profit,alloc,whalloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld_updated,ud_updated,tc,distances,max_alloc,stored_warehouse,dist_ware_cap_numpy,price_n,ld1,expensive_crop, allocation=0,optimize=1)
    print(Profit)
    
    # print(f"COP = {p}")
    # print(f'Revenue = {r}')
    # print(f'Transport cost = {t}')
    # print(f'Profit = {Profit}')


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
    plot_all_metrics(alloc,new_allocation_numpy,no_dist,no_crop,cost,crop_yield,price,ld_updated,ud_updated,tc,distances,max_alloc,stored_warehouse,dist_ware_cap_numpy,price_n,ld1,expensive_crop)
    
    print(whalloc)
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
    # plot_max_min(no_dist,no_crop,cost,crop_yield,price,tc,distances,max_alloc,standard_allocation_numpy,ld_updated,ud_updated,stored_warehouse,dist_ware_cap_numpy,price_n,ld1,expensive_crop)

if __name__ == '__main__':
    main()