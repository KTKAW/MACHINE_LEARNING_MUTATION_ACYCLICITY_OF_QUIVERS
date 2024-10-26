'''Cluster Algebra Functions used in proof of Proposition 2.8'''
#Import libraries 
#from sage import *
import numpy as np
import networkx as nx
from itertools import permutations
from copy import deepcopy as dc

# Function to convert matrix --> quiver
def Mat2Quiver(quiver_bmatrix):
    return nx.from_numpy_matrix(np.clip(quiver_bmatrix,0,None), create_using=nx.DiGraph)


# Function to plot an input quiver matrix
def QuiverPlot(quiver_bmatrix):
    test_quiver = Mat2Quiver(quiver_bmatrix)
    pos = nx.circular_layout(test_quiver)
    nx.draw_networkx(test_quiver,pos=pos,with_labels=True)
    nx.draw_networkx_edge_labels(test_quiver, pos=pos, edge_labels=nx.get_edge_attributes(test_quiver, 'weight'), label_pos=0.35)
    
    
# Function to create a 4x4 antisymmetric matrix
def create_antisymmetric_matrix(a, b, c, d, e, f):
    matrix = np.array([
        [ 0,  a,  b,  c],
        [-a,  0,  d,  e],
        [-b, -d,  0,  f],
        [-c, -e, -f,  0]
    ])
    return matrix


# Function to check whether a quiver matrix is weakly connected
def matrix_weakly_connected_check(matrix):
    # Convert the matrix to a quiver
    quiver = Mat2Quiver(matrix)
    # Create the undirected version of the quiver
    quiver_undirected = quiver.to_undirected()
    return nx.is_connected(quiver_undirected)    
  
  
# Function to define the canonical form of a matrix (lexicographically smallest)
triu_indices = np.triu_indices(4,k=1)
def find_canonical_form(matrix):
    canonical_matrix = matrix
    perms = permutations(range(4)) #...4 is hardcoded here for speed, otherwise matrix.shape[0]
    next(perms) #...skip the identity transform
    for perm in perms:
        # Permute rows and columns
        permuted_matrix = matrix[np.ix_(perm, perm)]
        # Keep track of the lexicographically smallest matrix
        if list(permuted_matrix[triu_indices]) < list(canonical_matrix[triu_indices]):
            canonical_matrix = permuted_matrix
    return canonical_matrix
    

# Function to check if the matrix's graph is acyclic
def matrix_acyclic_check(matrix, strongly=True):
    # Create a directed graph from the matrix
    G = nx.from_numpy_matrix(np.clip(matrix,0,None), create_using=nx.DiGraph)
    
    if strongly:
        # Strongly acyclic check: Check if the graph is a Directed Acyclic Graph (DAG)
        return nx.is_directed_acyclic_graph(G)
    else:
        # Weakly acyclic check: Convert to undirected and check if it's a forest (no cycles)
        undirected_G = G.to_undirected()
        return nx.is_forest(undirected_G)


# Function to check if a graph contains a directed cycle where each edge has multiplicity 2
def contains_markov_quiver(matrix):
    # Create a directed graph from the matrix
    G = nx.from_numpy_matrix(np.clip(matrix,0,None), create_using=nx.DiGraph)
    
    try:
        # Find all simple directed cycles in the graph
        cycles = list(nx.simple_cycles(G))
    except nx.NetworkXNoCycle:
        return False
    
    # Check each cycle to see if all edges have multiplicity 2
    for cycle in cycles:
        if len(cycle) != 3:
            continue
        has_multiplicity_2 = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]  # Next node in the cycle
            
            # Check if the edge weight is exactly 2 or -2
            if abs(matrix[u][v]) != 2:
                has_multiplicity_2 = False
                break
        
        # If we find a cycle where all edges have multiplicity 2, return True
        if has_multiplicity_2:
            return True
    
    # No such cycle found
    return False
    

#Function to mutate a quiver matrix at a specified node
def matrix_mutation(matrix, k):
    mutated_matrix = dc(matrix) 
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == k or j == k: 
                mutated_matrix[i,j] = -matrix[i,j]
            elif matrix[i,k]*matrix[k,j] > 0: 
                ###need an explicit check of this being integer (value == np.round(value))?
                mutated_matrix[i,j] = int(matrix[i,j] + matrix[i,k]*matrix[k,j]*(matrix[i,k]/abs(matrix[i,k])))
    return mutated_matrix


#Function to generate a matrix's quiver exchange graph
def ExchangeGraph(init_seed,depth,max_em=np.inf):  #...input inital seed as a numpy array, and the max depth to mutate to
    n = init_seed.shape[0]         #...extract number of mutable variables
    seed_list, current_seeds, next_seeds, checkable_seeds = [init_seed], [], [[init_seed,0,-1]], [0,0]
    #Mutate initial seed to specified depth
    for d in range(1,depth+1):
        #Update the current seeds to mutate, and the seeds in previous 2 depths to check against
        current_seeds = next_seeds
        next_seeds = []
        checkable_seeds[0] = checkable_seeds[1]                #...update list position of start of 2 depths ago (only need to check these for equivalence)
        checkable_seeds[1] = len(seed_list)-len(current_seeds) #...save end of last depth position for next iteration
        #Loop through the current seeds, mutating all their other variables
        for seed_info in current_seeds:
            #Identify the nodes to mutate about (if first depth then mutate them all)
            if d == 1: variable_list = list(range(n))
            else: variable_list = list(range(n))[:seed_info[2]]+list(range(n))[seed_info[2]+1:] #...skip last vertex mutated about
            #Mutate about all nodes not previously mutated about
            for variable in variable_list: 
                new_seed = matrix_mutation(seed_info[0],variable)
                #Skip if the new quiver has too large an edge multiplicity 
                if np.max(np.absolute(new_seed)) > max_em:
                    continue
                new_test = True              #...boolean to verify if the generated seed is new
                #Loop through previous 2 depths in EG, check if the new seed matches any of them (earlier depths have already been exhaustively mutated)
                new_seed_canonical = find_canonical_form(new_seed)
                for old_seed in seed_list[checkable_seeds[0]:]: 
                    if np.all(find_canonical_form(old_seed) == new_seed_canonical):
                        new_test = False    
                        break   
                if new_test:
                    seed_list.append(new_seed)
                    next_seeds.append([new_seed,len(seed_list)-1,variable]) #...save new seed to mutate, the seed's label in the EG, and the cluster variable to not mutate in next iteration
        #Check if any new seeds were generated at this depth, otherwise terminate loop
        if len(next_seeds) == 0:
            break
            
    return seed_list    


# Function to search quiver exchange graph to find acyclic or NMA-equivalence
def EG_Search_MAChecker(init_seed,depth,NMA_mutate_set):  #...input inital seed as a numpy array, and the max depth to mutate to
    n = init_seed.shape[0]         #...extract number of mutable variables
    variable_list = list(range(n))
    seed_list = [[],[],[[init_seed,0,-1]]]
    #Mutate initial seed to specified depth
    for d in range(1,depth+1):
        seed_list = seed_list[1:]
        seed_list[0] = [find_canonical_form(seed_info[0]) for seed_info in seed_list[0]] #...put in canonical form and remove vertex info
        seed_list.append([])
        #Loop through the current seeds, mutating all their other variables
        for seed_info in seed_list[-2]:
            #Identify the nodes to mutate about (if first depth then mutate them all)
            if d > 1: variable_list = list(range(n))[:seed_info[1]]+list(range(n))[seed_info[1]+1:] #...skip last vertex mutated about
            #Mutate about all nodes not previously mutated about
            for variable in variable_list: 
                new_seed = matrix_mutation(seed_info[0],variable)
                #Check for acyclicity
                if matrix_acyclic_check(new_seed,strongly=True):
                    return True
                #Check for the Markov quiver
                if NMA_checker(new_seed):
                    print('NMA_checker True...')
                    return False
                
                #Loop through all previous depth and new seeds in EG, check if the new seed matches any of them
                new_test = True #...boolean to verify if the generated seed is new
                new_seed_canonical = find_canonical_form(new_seed)
                #Check 2 depths back
                for old_seed in seed_list[0]: #...these already in canonical form
                    if np.all(old_seed == new_seed_canonical):
                        new_test = False    
                        break   
                #Check 1 depth back (which is being mutated now)
                if new_test:
                    for old_seed in seed_list[1]: 
                        if np.all(find_canonical_form(old_seed[0]) == new_seed_canonical):
                            new_test = False    
                            break  
                #Check current depth (which is being produced)
                if new_test:
                    for old_seed in seed_list[2]: 
                        if np.all(find_canonical_form(old_seed[0]) == new_seed_canonical):
                            new_test = False    
                            break  
    
                if new_test:
                    seed_list[-1].append([new_seed,variable])
                
        #Check if any new seeds were generated at this depth, otherwise terminate loop
        if len(seed_list[-1]) == 0:
            print('exhausted EG...',flush=True)
            return False #...here the full exchange graph has been produced, with no acyclic quivers!
            
        #Check for isomorphism to the NMA_mutate_set set
        for matrix in seed_list[-1]: #...previous depths already checked
            for nma_mutate_matrix in NMA_mutate_set:
                matrix_canonical = find_canonical_form(matrix[0])
                if np.all(matrix_canonical == nma_mutate_matrix):
                    print('NMA-isomorphism match...')
                    return False
            
        #Output progress     
        print(d,list(map(len,seed_list)),flush=True)
    
    #If reach max depth and nothing found, return none
    return None 


# Function to check for general non-mutation-acyclicity properties
def NMA_checker(matrix):
    # Create a directed graph from the matrix
    G = nx.from_numpy_matrix(np.clip(matrix,0,None), create_using=nx.DiGraph)
    
    # Loop through all sets of three nodes u, v, w
    for u in G.nodes:
        for v in G.successors(u):
            if v == u:
                continue
            #if abs(G[u][v].get('weight', None)) < 2:
            #    continue
            for w in G.successors(v):
                if w == u or w == v:
                    continue
                # Check if there's an edge from w to u to complete the cycle
                if G.has_edge(w, u):
                    # Get the weights of the edges in the cycle (automatically +ve)
                    weight_uv = G[u][v].get('weight', None)
                    weight_vw = G[v][w].get('weight', None)
                    weight_wu = G[w][u].get('weight', None)
                        
                    # Compute markov constant
                    C = weight_uv**2 + weight_vw**2 + weight_wu**2 - weight_uv*weight_vw*weight_wu
                    if C < 0:
                        return True
                    
                    if C <= 4 and weight_uv >=2 and weight_vw >= 2 and weight_wu >= 2:
                        return True

                    # Check if all weights are the same 
                    if weight_uv < 2 or weight_vw < 2 or weight_wu < 2: 
                        continue
                    if weight_uv == weight_vw == weight_wu:
                        return True
                    
    # If no such cycle is found, return False
    return False
