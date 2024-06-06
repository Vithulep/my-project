import os
import os.path as osp
import argparse
import readfile
from allocation import Allocation
from import_export import Import_export
from transportation import Greedy
from numpy.random import randint
from numpy.random import rand
import time
import pandas as pd
import csv 

def calc_distances(districts):
    distances={}
    for d1 in districts.values():
        for d2 in districts.values():
            if d1.id != d2.id:
                distances[(d1.id,d2.id)]=d1.coordinates.distance(d2.coordinates)
    distances = dict(sorted(distances.items(), key=lambda item: item[1]))
    return distances

def hyperparameter_tuner(objective, others, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    n_bits_list = [4,8,16,32,64]
    n_iter_list = [100,200,500,1000]
    n_pop_list = [10,20,50,100,200,500]
    r_cross_list = [i/10 for i in range(1,10,1)]
    r_mut_list = [1.0 / (float(n_bits) * len(bounds))*i for i in [0.01,0.1,1,10,100]]
    
    print("Bits expt")
    bits_res=[]
    for i in n_bits_list:
        print(f"bit: {i}")
        start=time.time()
        best, score = genetic_algorithm(objective, others, bounds, i, n_iter, n_pop, r_cross, r_mut)
        end=time.time()
        bits_res.append([i,score,(end-start)/60])
    fields = ['Bits', 'Profit', 'Time'] 
    with open('bits.csv', 'w') as f:
        write = csv.writer(f) 
        write.writerow(fields)
        write.writerows(bits_res)
    
    print("Iter expt")
    iter_res=[]
    for i in n_iter_list:
        print(f"iter: {i}")
        start=time.time()
        best, score = genetic_algorithm(objective, others, bounds, n_bits, i, n_pop, r_cross, r_mut)
        end=time.time()
        iter_res.append([i,score,(end-start)/60])
    fields = ['Iterations', 'Profit', 'Time'] 
    with open('iter.csv', 'w') as f:
        write = csv.writer(f) 
        write.writerow(fields)
        write.writerows(iter_res)
    
    print("Pop expt")
    pop_res=[]
    for i in n_pop_list:
        print(f"pop: {i}")
        start=time.time()
        best, score = genetic_algorithm(objective, others, bounds, n_bits, n_iter, i, r_cross, r_mut)
        end=time.time()
        pop_res.append([i,score,(end-start)/60])
    fields = ['Pop_size', 'Profit', 'Time'] 
    with open('pop.csv', 'w') as f:
        write = csv.writer(f) 
        write.writerow(fields)
        write.writerows(pop_res)
    
    print("Cross expt")
    cross_res=[]
    for i in r_cross_list:
        print(f"cross: {i}")
        start=time.time()
        best, score = genetic_algorithm(objective, others, bounds, n_bits, n_iter, n_pop, i, r_mut)
        end=time.time()
        cross_res.append([i,score,(end-start)/60])
    fields = ['cross_rate', 'Profit', 'Time'] 
    with open('cross.csv', 'w') as f:
        write = csv.writer(f) 
        write.writerow(fields)
        write.writerows(cross_res)
    
    print("Mut expt")
    mut_res=[]
    for i in r_mut_list:
        print(f"mutation: {i}")
        start=time.time()
        best, score = genetic_algorithm(objective, others, bounds, n_bits, n_iter, n_pop, r_cross, i)
        end=time.time()
        mut_res.append([i,score,(end-start)/60])
    fields = ['Mut_rate', 'Profit', 'Time'] 
    with open('mut.csv', 'w') as f:
        write = csv.writer(f) 
        write.writerow(fields)
        write.writerows(mut_res)

            

def allocation_to_csv(allocation):
    data = ['District ID']
    index = [(i+1) for i in range(len(allocation.allocation.keys()))]
    for i in allocation.allocation.keys():
        data.append(allocation.allocation[i])
    df = pd.DataFrame(data, index)
    df.to_csv('out.csv')
    

def parse_args():
    arg_parser = argparse.ArgumentParser()

    # input argument options
    arg_parser.add_argument(dest='example_path',
                            type=str,
                            help='Pass the path to the data folder')
    args = arg_parser.parse_args()
    return args

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop)) #randit b/w 0-500 
	for ix in randint(0, len(pop), k-1):    
		# check if better (e.g. perform a tournament)
		if scores[ix] > scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded

def objective(standard_allocation, new_allocation, import_export, districts, crops, compliance, distances):
    
    #checking for allocation feasibility 
    # for district in districts.values():
    #     sum1,sum2 = 0,0
    #     for crps in list(crops.keys()):
    #         sum1 += standard_allocation.allocation[district.id][crps]
    #         sum2 += new_allocation.allocation[district.id][crps]  
    #     if sum2>sum1:
    #           return 0
    

    allocation=Allocation(standard_allocation.allocation,new_allocation.allocation,compliance)
    # import_export_obj = Import_export(allocation.allocation,import_export.allocation)
    # import_export_obj.do_import_export()
    for district in districts.values():
        district.produce(allocation,crops)
    transportation = Greedy(districts,crops,distances)
    flag,feasibility = transportation.solution_feasible(districts,crops)
    if not flag:
        return 0
    
    transportation.start_transportation(districts,crops)
    total_revenue=0
    total_production_cost=0
    for district in districts.values():
        total_revenue+=district.get_revenue(crops)
        total_production_cost+=district.production_cost

    return round(total_revenue-total_production_cost-transportation.cost,2)

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded

# genetic algorithm
def genetic_algorithm(objective, others, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    standard_allocation = others[0]
    new_allocation = others[1]
    import_export = others[2]
    districts = others[3]
    crops = others[4]
    compliance = others[5]
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
	# keep track of best solution- Converts bitstings to allocation for Karnataka
    decoded = decode(bounds, n_bits, pop[0])
    i=0
    for district in districts.values():
                for crps in list(crops.keys()):
                    new_allocation.allocation[district.id][crps]=decoded[i]
                    i+=1
    
    #Making allocations feasible: Area constraint
    for district in districts.values():
        sum1 = 0
        sum2 = 0
        for crps in list(crops.keys()):
            sum1 += standard_allocation.allocation[district.id][crps]
            sum2 += new_allocation.allocation[district.id][crps]
        for crps in list(crops.keys()):
            new_allocation.allocation[district.id][crps] = (new_allocation.allocation[district.id][crps]/sum2)*sum1
    
    distances = calc_distances(districts)
    best, best_eval = 0, objective(standard_allocation, new_allocation, import_export, districts, crops, compliance, distances)
    
    # enumerate generations
    for gen in range(n_iter):
        if gen%20==0:
            print(f"iteration no : {gen}")
        # decode population
        decoded = [decode(bounds, n_bits, p) for p in pop]
        # evaluate all candidates in the population
        scores = []
        for d in decoded:
            i=0
            for district in districts.values():
                        for crps in list(crops.keys()):
                            new_allocation.allocation[district.id][crps]=d[i]
                            i+=1
            for district in districts.values():
                sum1 = 0
                sum2 = 0
                for crps in list(crops.keys()):
                    sum1 += standard_allocation.allocation[district.id][crps]
                    sum2 += new_allocation.allocation[district.id][crps]
                for crps in list(crops.keys()):
                    new_allocation.allocation[district.id][crps] = (new_allocation.allocation[district.id][crps]/sum2)*sum1
    
            scores.append(objective(standard_allocation, new_allocation, import_export, districts, crops, compliance, distances))
        # check for new best solution
        
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                # print(">%d, new best = %f" % (gen, scores[i]))
        
        # select parents: G*P
        selected = [selection(pop, scores) for _ in range(n_pop)]
        
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]

def main():
    st = time.time()
    args = parse_args()
    example_path = args.example_path
    if example_path[-1]!='/':
        example_path+='/'
    #Reading the inputs
    districts=readfile.read_districts(example_path,'district.csv')
    crops = readfile.read_crops(example_path,'crop.csv')
    new_allocation = readfile.read_allocation(example_path,'new_allocation.csv',crops)
    standard_allocation=readfile.read_allocation(example_path,'standard_allocation.csv',crops)
    import_export=readfile.read_import_export(example_path,'import_export.csv',crops)
    compliance=readfile.read_parameters(example_path,'parameters.csv')
    
    # define range for input
    n_bounds = (len(list(crops.keys())) * len(districts.values()))
    #Bound for distict wise crop wise land allocation
    bounds = [[0, 1000000.0]] * (n_bounds)

    #Hyper-parameters 


    # define the total iterations
    n_iter = 1000
    # bits per variable
    n_bits = 32
    # define the population size
    n_pop = 500
    # crossover rate
    r_cross = 0.9
    # mutation rate
    r_mut = 1.0 / (float(n_bits) * len(bounds))
    # r_mut = 0.0000520833333333333

    
    others = [standard_allocation, new_allocation, import_export, districts, crops, compliance]
    
    # hyperparameter_tuner(objective, others, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
    
    # ----------perform the genetic algorithm search------------
    best, score = genetic_algorithm(objective, others, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
    print('Done!')
    et = time.time()
    # get execution time in minutes
    res = et - st
    final_res = res / 60
    print('Total Execution time:', final_res, 'minutes')
    # print(best, score)
    decoded = decode(bounds, n_bits, best)
    i=0
    for district in districts.values():
                for crps in list(crops.keys()):
                    new_allocation.allocation[district.id][crps]=decoded[i]
                    i+=1
    print(f"decoded:{decoded}")
    print(f"Score: {score}")
    print(f"Allocation: {new_allocation.allocation}")
    #-----------------------------------------------------------

    # allocation_to_csv(new_allocation)
    et = time.time()
    # get execution time in minutes
    res = et - st
    final_res = res / 60
    print('Total Execution time:', final_res, 'minutes')
    

if __name__ == '__main__':
    main()
