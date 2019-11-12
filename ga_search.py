import numpy as np
from time import  time
from random import seed,uniform,randint
from creat_data import creat_cnf
from compute_acc import acc_compute
import numba
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--size_of_X', '-x', default=300, type = int)
parser.add_argument('--size_of_C', '-c', default=900, type = int)
parser.add_argument('--k_sat', '-k', default=3, type = int)
parser.add_argument('--population_num', '-p', default=800, type = int)
parser.add_argument('--cross_rate', '-cr', default=0.9, type =float)
parser.add_argument('--mutate_rate', '-mr', default=5, type = int)
parser.add_argument('--keep_rate', '-kr', default=0.8, type = float)
args = parser.parse_args()

print(args)

size_of_X=args.size_of_X
size_of_C=args.size_of_C
k_sat=args.k_sat
population_num=args.population_num #必须是2的倍数
cross_rate=args.cross_rate
mutate_rate=args.mutate_rate   #n/1000中的n

keep_rate=args.keep_rate


start_time=time()

def population_init(population_num):

	X, C_np = creat_cnf(size_of_X, size_of_C, k_sat, population_num)
	#print(X, C_np)
	#print(X.shape, C_np.shape)
	population=X
	SAT=C_np

	return population,SAT
@numba.jit
def fitness_fn(population,population_num):

	fitness_list=[]
	for i in range(population_num):
		fitness_list.append(acc_compute(size_of_C,k_sat,SAT,population[i]))

	return fitness_list
@numba.jit
def cross_fn(parent1,parent2):
	#child1=[]
	#child2=[]
	#print(parent1[:int(len(parent1)*cross_rate),])
	#print(parent2[int(len(parent1)*cross_rate):,])
	child1=parent1[:int(len(parent1)*cross_rate)]+parent2[int(len(parent1)*cross_rate):]
	child2 = parent2[:int(len(parent1) * cross_rate)] +parent1[int(len(parent1) * cross_rate):]
	#print(child1.type)
	child1=mutate_fn(child1)
	child2=mutate_fn(child2)
	return child1,child2
@numba.jit
def mutate_fn(one_of_population):
	for i in range(len(one_of_population)):
		if randint(1,1000)<=mutate_rate:
			one_of_population[i]=(one_of_population[i]+1)%2
	return one_of_population
@numba.jit
def make_new_population(population_num,population,which_one,new_population_num):
	new_population=[]

	for i in range(int((population_num)*keep_rate)//2):
		parent1=population[which_one]
		parent2=population[randint(0,population_num-1)]
		child1,child2=cross_fn(parent1,parent2)
		new_population.append(child1)
		new_population.append(child2)
	for i in range((population_num-int((population_num)*keep_rate))//2):
		parent1=population[randint(0,population_num-1)]
		parent2=population[randint(0,population_num-1)]
		child1,child2=cross_fn(parent1,parent2)
		new_population.append(child1)
		new_population.append(child2)

	if len(new_population)<new_population_num:
		for i in range((population_num - int((population_num) * keep_rate)) // 2):
			parent1 = population[randint(0, population_num - 1)]
			parent2 = population[randint(0, population_num - 1)]
			child1, child2 = cross_fn(parent1, parent2)
			new_population.append(child1)
			new_population.append(child2)
	if len(new_population)>new_population_num:
		return new_population[:new_population_num]
	return new_population

gen_round=0

population,SAT=population_init(population_num)
fitness_list=fitness_fn(population,population_num)
fitness_best=max(fitness_list)
fit_history=fitness_best
which_one=[i for i in range(population_num) if fitness_list[i]==fitness_best][0]
same_times=0
print('gen:',gen_round,'   fitness_best=',fitness_best)


new_population_num=population_num

while fitness_best!=1.0:

	t1=time()
	population=make_new_population(population_num,population,which_one,new_population_num)
	population_num=new_population_num
	#print()
	gen_round+=1

	fitness_list = fitness_fn(population, population_num)
	fitness_best = max(fitness_list)
	if fit_history==fitness_best:
		same_times+=1
	else:
		same_times=0
	fit_history=fitness_best
	if same_times>10:
		mutate_rate+=5
		keep_rate-=0.01
		cross_rate-=0.01
#		population_num+=10
		
		print('mr up to:',mutate_rate)
	if keep_rate<=0.2:
		keep_rate=0.5
	if cross_rate<=0.3:
		cross_rate=0.5
	if mutate_rate>=1000:
		mutate_rate=500

	which_one = [i for i in range(population_num) if fitness_list[i] == fitness_best][0]
	if same_times>10:
		new_population_num=population_num+10
		same_times=0
	t2=time()
	print('gen:', gen_round, '   fitness_best=', fitness_best,'   spend_time=',t2-t1)

end_time=time()
print(population[which_one])
print('cost:',end_time-start_time)

