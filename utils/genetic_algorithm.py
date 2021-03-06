import random

# create a binary string of length num_vars
# we have many variables, but only want a select few to enter our model
def create_binary_string(num_vars, max_vars):
    """
    binary_string=""
    for i in range(num_vars):
        binary_string+=str(random.randint(0,1))
    return binary_string
    """
    binary_string=list("0"*num_vars)
    random_spot=0
    num_ones=random.randint(0,max_vars)
    for i in range(num_ones+1):
        random_spot=random.randint(0,num_vars-1)
        binary_string[random_spot]='1'
     
    return "".join(binary_string)
        
# pass a list of binary strings with their accompanying fitness ratio as a tuple pair
# and return a tuple pair of selected chromosomes 
def select_chromosome_pair(chromosome_fitness_ratio_pairs): 
    sorted_by_fitness_ratio=sorted(chromosome_fitness_ratio_pairs, key=lambda x: x[1])
    fitness_ratios=[pair[1] for pair in sorted_by_fitness_ratio]
    accumulated_fitness_ratios=[sum(fitness_ratios[:i]) for i in range(1,len(fitness_ratios)+1)]
    chromosome1=[]
    chromosome2=[]
    while chromosome1==chromosome2:
        rand_num1=random.random()
        rand_num2=random.random()
        for i in range(len(sorted_by_fitness_ratio)):
            if rand_num1<accumulated_fitness_ratios[i]:
                chromosome1=sorted_by_fitness_ratio[i][0]
                break
                
        for i in range(len(sorted_by_fitness_ratio)):
            if rand_num2<accumulated_fitness_ratios[i]:
                chromosome2=sorted_by_fitness_ratio[i][0]
                break      
            
    return (chromosome1,chromosome2)

# takes a tuple pair of chromosomes, split them if they are chosen to mate and rejoins
# them to create two new chromosomes. Each new child is then subject to mutation of
# a single bit if chosen to mutate.
def mate_pair(pair, crossover_probability, mutation_probability):
    rand_num1=random.random()
    rand_num2=random.random()
    chromosome_length=len(pair[0])
    crossover_point=random.randint(1,chromosome_length-1)
    mutation_point=random.randint(0,chromosome_length-1)
    
    new_chromosome1=""
    new_chromosome2=""
    
    if rand_num1<crossover_probability:
        new_chromosome1=pair[0][:crossover_point] + pair[1][crossover_point:]    
        new_chromosome2=pair[1][:crossover_point] + pair[0][crossover_point:]    
    else:
        # create clones
        new_chromosome1=pair[0]
        new_chromosome2=pair[1]
    
    if rand_num2<mutation_probability:
        if new_chromosome1[mutation_point]=='1':
            new_chromosome1=list(new_chromosome1)
            new_chromosome1[mutation_point]='0'
            new_chromosome1="".join(new_chromosome1)
        else:
            new_chromosome1=list(new_chromosome1)
            new_chromosome1[mutation_point]='1'
            new_chromosome1="".join(new_chromosome1)
            
        if new_chromosome2[mutation_point]=='1':
            new_chromosome2=list(new_chromosome2)
            new_chromosome2[mutation_point]='0'
            new_chromosome2="".join(new_chromosome2)
        else:
            new_chromosome2=list(new_chromosome2)
            new_chromosome2[mutation_point]='1'
            new_chromosome2="".join(new_chromosome2)

    return (new_chromosome1,new_chromosome2)
            
# produce list of binary strings that represent chromosomes            
def generate_initial_population(chromosome_length, population_size, max_vars):
    chromosome_population=[]
    for i in range(population_size):
        chromosome_population.append(create_binary_string(chromosome_length, max_vars))
    return chromosome_population    