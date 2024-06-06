import os
import os.path as osp
import argparse
import readfile
from allocation import Allocation
from import_export import Import_export
from transportation import Greedy
import csv


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
    distances={}
    for d1 in districts.values():
        for d2 in districts.values():
            if d1.id != d2.id:
                distances[(d1.id,d2.id)]=d1.coordinates.distance(d2.coordinates)
    distances = list(distances.items())
    # print(type(int(distances[0][0][0])))
    result = sorted(distances, key=lambda x: (int(x[0][0]),x[1]))
    # for i,j in zip(result,range(len(result))):
    #     print(f"{i} =====> {j}")
    i=0
    j=0
    list1 = []
    list2 = []
    for i in range(len(result)):
        if i%29 == 0:
            if i>0:
                list1.append(list2)
                list2 = []
            list2.append(districts[result[i][0][0]].name)
            list2.append(districts[result[i][0][1]].name)
        else:
            list2.append(districts[result[i][0][1]].name)
    list1.append(list2)
    with open("district_dist.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(list1)
            








if __name__ == '__main__':
    main()



