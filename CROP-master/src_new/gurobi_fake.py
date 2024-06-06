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
        plt.plot(x_labels,avg_dict[state])         
        #-----------------Bar graph----------------------
        # labels = []
        # for x in range(11):
        #     labels.append((x+i*0.2-0.4)/10)
        # plt.bar(labels, avg_dict[state], align='edge', width = barWidth)

    plt.title(name)
    plt.legend(list(avg_dict.keys()), shadow=True)
    plt.ylabel('In billion Rupees')
    plt.xlabel('Compliance ratio')

def parse_args():
    arg_parser = argparse.ArgumentParser()

    # input argument options
    arg_parser.add_argument(dest='example_path',
                            type=str,
                            help='Pass the path to the data folder')
    args = arg_parser.parse_args()
    return args

def plot_fake_alloc(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,current_allocation_numpy,allocation=0,optimize=1):
    d={'Revenue':[],'Production Cost':[],'Transportation Cost':[],'Profit':[]}
    for c in range(11):
        print(f'---------------------{c/10}-----------------------')
        p,r,t,Profit,alloc,fake_alloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,c/10,current_allocation_numpy,allocation,optimize)
        d['Revenue'].append(r/1e9)
        d['Production Cost'].append(p/1e9)
        d['Transportation Cost'].append(t/1e9)
        d['Profit'].append(Profit/1e9)
    plot_results("Revenue, Costs and Profit vs Compliance",d)
    plt.show()

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

    return district_numpy,crops_numpy,dist_crop_numpy,standard_allocation_numpy,new_allocation_numpy

def allocation_to_csv(allocation):
    data = []
    index = [(i+1) for i in range(len(allocation.allocation.keys()))]
    # print(index)
    for i in allocation.allocation.keys():
        data.append(allocation.allocation[i])
    # print(f'len data:{data}')
    df = pd.DataFrame(data, index)
    df.to_csv('allocation1.csv')

def print_to_file(districts,crops,alloc,new_allocation):
    for district,i in zip(districts.values(),range(len(districts.values()))):
                for crps,j in zip(list(crops.keys()),range(len(crops.keys()))):
                    new_allocation.allocation[district.id][crps]=alloc[i,j]
    print(f"Allocation: {new_allocation.allocation}")
    allocation_to_csv(new_allocation)

def plot_all_metrics(alloc1,current_allocation_numpy,no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc):
    recommended_allocation = alloc1
    d={'Revenue':[],'Production Cost':[],'Transportation Cost':[],'Profit':[]}
    for c in range(11):
        print(f'---------------------{c/10}-----------------------')
        allocation = (1-c/10)*current_allocation_numpy + (c/10)*recommended_allocation
        p,r,t,Profit,alloc,fake_alloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,c/10,current_allocation_numpy,allocation,optimize=0)
        d['Revenue'].append(r/1e9)
        d['Production Cost'].append(p/1e9)
        d['Transportation Cost'].append(t/1e9)
        d['Profit'].append(Profit/1e9)
    plot_results("Revenue, Costs and Profit vs Compliance ratio",d)
    plt.show()

def plot_max_min(no_dist,no_crop,cost,crop_yield,price,tc,distances,max_alloc,standard_allocation_numpy,ld,ud):
    for i in range(5):
        ld_factor = 0.9-0.1*i
        ud_factor = 1.1+0.1*i
        ld_temp = ld
        ud_temp = ud
        ld_temp[:,0] = ld[:,0]*ld_factor/0.8
        ud_temp[:,0] = ud[:,0]*ud_factor/1.2
        p,r,t,Profit,alloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld_temp,ud_temp,tc,distances,max_alloc,allocation=0,optimize=1)
        profit_vector = profit_complaince(alloc,standard_allocation_numpy,no_dist,no_crop,cost,crop_yield,price,ld_temp,ud_temp,tc,distances,max_alloc)
        plt.plot(profit_vector,label=str(round(ld_factor,1))+','+str(round(ud_factor,1)))
    plt.xticks(list(range(11)),map(lambda x: x/10,list(range(11))))
    plt.ylabel('Profit in INR')
    plt.xlabel('Compliance')
    plt.legend()
    plt.show()

#Calculates profit for different compliance values
def profit_complaince(alloc1,standard_allocation_numpy,no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc):
    recommended_allocation = alloc1
    profit_vector = []
    for c in range(11):
        print(f'---------------------{c/10}-----------------------')
        allocation = (1-c/10)*standard_allocation_numpy + (c/10)*recommended_allocation
        p,r,t,Profit,alloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,allocation,optimize=0)
        profit_vector.append(Profit)
    return profit_vector

def gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,LD,UD,TC,distances,max_alloc,C,standard_allocation,allocation,optimize):
    '''
    allocation -> for finding the total cropped area
    index i is for districts and j is for crops 
    '''

    #----------------------Gurobi code----------------------
    model = Model(name='CROP')
    if optimize == 1:
        allocation = model.addVars(no_dist,no_crop,vtype=GRB.CONTINUOUS,lb = 0, name = 'allocation')  #For allocation
        fake_allocation = model.addVars(no_dist,no_crop,vtype=GRB.CONTINUOUS,lb = 0, name = 'fake_allocation')  #For fake_allocation
        model.addConstrs(allocation[i,j] == (1-C)*standard_allocation[i,j]+C*fake_allocation[i,j] for i in range(no_dist) for j in range(no_crop) )

    x = model.addVars(no_dist,no_crop*no_dist,vtype=GRB.CONTINUOUS,lb = 0, name = 'x')  #For transportation 
    
    #----------------------------------Adding constarints to gurobi model------------------------------------------------
    #Split these constraint equations
    
    # Checking for lower demand in each district  
    model.addConstrs((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i]\
                    for k in range(no_dist)))>=LD[i,j] for i in range(no_dist) for j in range(no_crop))
    
    # Checking for upper demand in each district
    model.addConstrs((allocation[i,j]*crop_yield[i,j]-sum(x[i,j*no_dist+k] for k in range(no_dist))+sum(x[k,j*no_dist+i]\
                    for k in range(no_dist)))<=UD[i,j] for i in range(no_dist) for j in range(no_crop))
    if optimize == 1: 
        model.addConstrs((sum(allocation[i,j] for j in range(no_crop))<=max_alloc[i]) for i in range(no_dist))
        # model.addConstrs((sum(fake_allocation[i,j] for j in range(no_crop))<=max_alloc[i]) for i in range(no_dist))


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
    alloc = allocation          #For initializing same shape as allocation
    fake_alloc = allocation         #For initializing same shape as allocation
    if model.status == GRB.Status.OPTIMAL:
        values = model.getAttr("X", model.getVars())
        values = np.array(values)
        if optimize == 1:
            alloc = values[0:no_dist*no_crop].reshape((no_dist,no_crop))
            fake_alloc = values[no_dist*no_crop:2*no_dist*no_crop].reshape((no_dist,no_crop))
            transport_qty = values[2*no_dist*no_crop:].reshape((no_dist,no_dist*no_crop))
            # print(f'Allocation: \n {alloc}')
            # print(f'Transported quantity: \n {transport_qty}')
            # np.savetxt('transport.csv', transport_qty, delimiter=',')
            # stock = np.zeros(alloc.shape)               #Final Stock
            # for i in range(no_dist):
            #     for j in range(no_crop):
            #         stock[i,j] = alloc[i,j]*crop_yield[i,j]-sum(transport_qty[i,j*no_dist+k] for k in range(no_dist))+sum(transport_qty[k,j*no_dist+i] for k in range(no_dist))
            # print(f'Stock: \n {stock}')
        else:
            transport_qty = values.reshape((no_dist,no_dist*no_crop))
        p = sum(alloc[i,j]*cost[i,j] for i in range(no_dist) for j in range(no_crop))
        r = sum((alloc[i,j]*crop_yield[i,j]-sum(transport_qty[i,j*no_dist+k] for k in range(no_dist))+sum(transport_qty[k,j*no_dist+i] for k in range(no_dist)))*price[i,j] for i in range(no_dist) for j in range(no_crop))
        t = sum((TC[j]*distances[i,k]*transport_qty[i,j*no_dist+k]) for k in range(no_dist) for i in range(no_dist) for j in range(no_crop))
        Profit = r-p-t
    return p,r,t,Profit,alloc,fake_alloc

#---------------------------------------Gurobi-----------------------------------------
def main():
    # get the start time
    st = time.time()
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'
    
    district_numpy,crops_numpy,dist_crop_numpy,standard_allocation_numpy,new_allocation_numpy = read_files(example_path)
    
    # For now area under cultivation is taken as past years allocation
    new_alloc = standard_allocation_numpy
    # Finding total cultivation area which would be maximum croped area districtwise 
    max_alloc = np.sum(new_alloc,axis=1)


    no_dist= district_numpy.shape[0]
    no_crop = crops_numpy.shape[0]


    distances = calc_distances(no_dist,district_numpy)

    quintal_to_ton = 10
    cost = dist_crop_numpy[:,:,2]               #Cost per crop per ton
    crop_yield = dist_crop_numpy[:,:,3]         #Yeild tonnes per hectare
    price = dist_crop_numpy[:,:,4]*quintal_to_ton              #price in tonnes
    ld = dist_crop_numpy[:,:,6]*0.7/0.7                  #Minimum demand of district in tonnes (Originally 70% of district-wise production)
    ud = dist_crop_numpy[:,:,5]*1.3/1.1             #Maximum dwmand of district in tonnes (110% of district-wise production)
    tc = crops_numpy[:,1]                       #Transportation cost in rupees per ton per km
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
    C = 0.7
    p,r,t,Profit,alloc,fake_alloc = gurobi_optimizer(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,C,new_allocation_numpy,allocation=0,optimize=1)
    # print(f"COP = {p}")
    print(f'Revenue = {r}')
    # print(f'Transport cost = {t}')
    print(f'Profit = {Profit}')

    #---------------------------------Plotting the graph of profit at various complinace----------------------------
    # plot_fake_alloc(no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc,new_allocation_numpy,allocation=0,optimize=1)

    #----------------------------------writing allocation to csv-----------------------------------------------------
    # print_to_file(districts,crops,alloc,new_allocation)
    
    #-----------------------------------Plotting revenue,TC,COP,Profit----------------------------------------
    # plot_all_metrics(fake_alloc,new_allocation_numpy,no_dist,no_crop,cost,crop_yield,price,ld,ud,tc,distances,max_alloc)
    
    #----------------------------------Plotting profit for different max-min demands---------------------------------------
    # plot_max_min(no_dist,no_crop,cost,crop_yield,price,tc,distances,max_alloc,standard_allocation_numpy,ld,ud)

if __name__ == '__main__':
    main()