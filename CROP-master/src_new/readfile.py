import copy
import json
import os.path as osp
import random
import re
from csv import DictReader

from district import District
from crop import Crop
from allocation import Allocation
from warehouse import Warehouse
from dist_warehouse import dist_warehouse

def read_districts(example_path,filename='district.csv'):
    districts={}
    with open(example_path+filename, 'r', encoding='utf-8-sig') as read_obj: #with used to close after reading
        csv_dict_reader = DictReader(read_obj)  # Create a CSV DictReader object to read the CSV file
        csv_list = list(csv_dict_reader) # Convert the CSV data into a list of dictionaries
        n = len(csv_list)

        for row in csv_list:
            districts[row['District ID']]=District(row['District ID'],row['Name'],row['Latitude'],row['Longitude'],row['Population'])
    # print(districts)
    return districts

# Read warehouse data
def read_warehouse(example_path,filename='warehouse.csv'):
    warehouses={}
    with open(example_path+filename, 'r', encoding='utf-8-sig') as read_obj: #with used to close after reading
        csv_dict_reader = DictReader(read_obj)  # Create a CSV DictReader object to read the CSV file
        csv_list = list(csv_dict_reader) # Convert the CSV data into a list of dictionaries
        n = len(csv_list)
        a=0
        for row in csv_list:
            a=a+1
            # print("\n")
            warehouses[row['WHID']]=Warehouse(row['WHID'],row['WH Name'],row['District'],row['Capacity'], row['Cost In Rs'], row['Latitude'], row['Longitude'])
    # print(districts)
        # print(a)
    return warehouses

# read warehouses district wise 
def read_ware_dist(example_path,filename='Wh_Allo_dist_f.csv'):
    warehouses={}
    with open(example_path+filename, 'r', encoding='utf-8-sig') as read_obj: #with used to close after reading
        csv_dict_reader = DictReader(read_obj)  # Create a CSV DictReader object to read the CSV file
        csv_list = list(csv_dict_reader) # Convert the CSV data into a list of dictionaries
        n = len(csv_list)
        a=0
        for row in csv_list:
            a=a+1
            # print("\n")
            warehouses[row['District ID']]=dist_warehouse(row['District ID'],row['capacity'],row['Bajra'],row['Cotton'], row['Groundnut'], row['Jowar'], row['Maiz'], row['Moong'], row['Onion'], row['Ragi'], row['Rice'], row['Soyabean'], row['Tur'], row['Wheat'])
    # print(districts)
        # print(a)
    return warehouses

#read stored crop according to expensiveness in warehouse


def read_crops(example_path,filename='crop.csv'):
    crops={}
    with open(example_path+filename, 'r', encoding='latin-1') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        csv_list = list(csv_dict_reader)
        n = len(csv_list)

        for row in csv_list:
            crops[row['Crop ID']]=(Crop(row['Crop ID'],row['Name'],row['Filename'],float(row['Transport Cost'])*1))

    for crop in crops.values():
        csv_list=None
        with open(example_path+crop.filename, 'r', encoding='utf-8-sig') as read_obj:
            csv_dict_reader = DictReader(read_obj)
            csv_list = list(csv_dict_reader)
            n = len(csv_list)

        cost_dict={}
        yield_dict={}
        price_dict={}
        demand_max_dict={}
        demand_min_dict={}
        price_dict_n={}
        yield_dict_n={}
        # Risk_dist_n = {}
        # print(csv_list)
        # return csv_list
        for row in csv_list:
            district_id=row['District ID']
            cost_dict[district_id]=float(row['Cost'])
            yield_dict[district_id]=float(row['Yield'])
            price_dict[district_id]=float(row['Price'])
            demand_max_dict[district_id]=float(row['Demand Max'])#*(1.1/1.1)
            demand_min_dict[district_id]=float(row['Demand Min'])#*(0.7/0.7)
            yield_dict_n[district_id]=float(row['Risk'])
            price_dict_n[district_id]=float(row['Price_n'])
            # Risk_dist_n [district_id] = float(row['Risk'])
        
        # values_array = np.array(list(price_dict_n.values()))
        # print(price_dict_n.shape)    

        crop.add_district_details(cost_dict,yield_dict,price_dict,demand_max_dict,demand_min_dict,price_dict_n,yield_dict_n)
    # print('1.2,0.7')
    return crops

def read_allocation(example_path,filename,crops):
    with open(example_path+filename, 'r', encoding='latin-1') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        csv_list = list(csv_dict_reader)
        n = len(csv_list)

        allocation={}
        for row in csv_list:
            district_id=row['District ID']
            allocation[district_id]={}
            for crop_id in crops:
                # print(crop_id)
                allocation[district_id][crop_id] = float(row[crop_id])

        return Allocation(allocation)

def read_parameters(example_path,filename='parameters.csv'):
    with open(example_path+filename, 'r', encoding='latin-1') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        csv_list = list(csv_dict_reader)
        #Return the only row available
        for row in csv_list:
            return float(row['Compliance'])

def read_import_export(example_path,filename,crops):
    with open(example_path+filename, 'r', encoding='latin-1') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        csv_list = list(csv_dict_reader)
        n = len(csv_list)

        allocation={}
        for row in csv_list:
            district_id=row['District ID']
            allocation[district_id]={}
            for crop_id in crops:
                allocation[district_id][crop_id]=float(row[crop_id])

        return Allocation(allocation)

