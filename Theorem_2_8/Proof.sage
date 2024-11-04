'''Proof Methodology'''
# 1) Generate All Quivers with |em| <= 2
# 2) Reduce to connected
# 3) Reduce to 1 representative per isomorphism class
# 4) Filter out all acyclic from the set
# 5) Create a set of NMA cases: box, dreaded, Markov subquiver (copy fn to check), and filter these out
# 6) Mutate the NMA to generate a set of NMA with |em| <= 2
# 7) Filter out those corresponding to this set
# 8) Mutate the remaining set until is acyclic or matches an NMA

#Import libraries
import sys
import numpy as np
import networkx as nx
from itertools import product
from ProofFunctions import *

#Set proof hyperparameters
nma_set_mutate_depth         = 5  #int(sys.argv[1])
nma_set_max_edge_multiplcity = 30 #int(sys.argv[2])
final_check_depth            = 12 #int(sys.argv[3])
print(f'Hyperparams: {(nma_set_mutate_depth,nma_set_max_edge_multiplcity,final_check_depth)}')

# Define the mutation-acyclic (MA), and non-mutation-acyclic (NMA) lists
MA = []
NMA = []

#########################################################################################
#%% # Step (1): Generate all
print('\n# Step (1) #\n',flush=True)
# Set of values for each matrix entry
values = [-2, -1, 0, 1, 2]

# Generate all combinations of the 6 independent variables (a, b, c, d, e, f)
combinations = product(values, repeat=6)

# Generate all 4x4 antisymmetric matrices
antisymmetric_matrices = [create_antisymmetric_matrix(*combo) for combo in combinations]

# Print the total number of generated matrices
print(f'Total number of antisymmetric matrices: {len(antisymmetric_matrices)}',flush=True)
print(f'Counts for (MA,NMA): {(len(MA),len(NMA))}\n',flush=True)

#########################################################################################
#%% # Step (2): Reduce to weakly connected digraphs
print('\n# Step (2) #\n',flush=True)
weakly_connected_matrices = [matrix for matrix in antisymmetric_matrices if nx.is_connected(Mat2Quiver(matrix).to_undirected())]

# Print the total number of weakly connected matrices
print(f'Total number of weakly connected matrices: {len(weakly_connected_matrices)}',flush=True)
print(f'Counts for (MA,NMA): {(len(MA),len(NMA))}\n',flush=True)

#########################################################################################
#%% # Step (3): Reduce by isomorphism
print('\n# Step (3) #\n',flush=True)
# Convert weakly connected matrices to their canonical form, and remove repeats (isomorphism equivalences)
#...note some biasing here from the generation process
unique_matrices = [find_canonical_form(matrix) for matrix in weakly_connected_matrices]
unique_matrices = np.unique(unique_matrices,axis=0)

# Print the total number of unique matrices up to permutation
print(f'Total number of unique matrices up to isomorphism: {len(unique_matrices)}',flush=True)
print(f'Counts for (MA,NMA): {(len(MA),len(NMA))}\n',flush=True)

#########################################################################################
#%% # Step (4): Filter out acyclic
print('\n# Step (4) #\n',flush=True)
# Check for strong acyclicity
strongly_acyclic_matrices = []
not_strongly_acyclic_matrices = []
for matrix in unique_matrices:
    if matrix_acyclic_check(matrix, strongly=True):
        strongly_acyclic_matrices.append(matrix)
    else:
        not_strongly_acyclic_matrices.append(matrix)

# Output the results
print(f'Total strongly acyclic matrices: {len(strongly_acyclic_matrices)} --> MA',flush=True)
print(f'Total not strongly acyclic matrices: {len(not_strongly_acyclic_matrices)}',flush=True)

# Append the strongly acyclic quivers to the MA list
MA += strongly_acyclic_matrices
print(f'Counts for (MA,NMA): {(len(MA),len(NMA))}\n',flush=True)

#########################################################################################
#%% # Step (5): Filter out known NMA
print('\n# Step (5) #\n',flush=True)
# Check each not-strongly-acyclic matrix to see if contains the Markov quiver, and sort them into appropriate lists
contains_markov = []
does_not_contain_markov = []
for matrix in not_strongly_acyclic_matrices:
    if contains_markov_quiver(matrix):
        contains_markov.append(matrix)
    else:
        does_not_contain_markov.append(matrix)

# Output the results
print(f'Total matrices containing Markov quiver: {len(contains_markov)} --> NMA',flush=True)
print(f'Total matrices not containing Markov quiver: {len(does_not_contain_markov)}',flush=True)

# Append the Markov quivers to the NMA list
NMA += contains_markov

# Define the NMA standard quivers
box_quiver = np.array([[0,2,0,-2],[-2,0,2,0],[0,-2,0,2],[2,0,-2,0]])
box_quiver = find_canonical_form(box_quiver)
dreaded_torus_quiver = np.array([[0,1,1,-1],[-1,0,-1,2],[-1,1,0,-1],[1,-2,1,0]])
dreaded_torus_quiver = find_canonical_form(dreaded_torus_quiver)

# Append these to the NMA list
NMA += [box_quiver,dreaded_torus_quiver]

# Remove those equivalent to these up to isomorphism
not_box_or_dreaded = []
box_or_dreaded_counter = 0
for matrix in does_not_contain_markov:
    if np.all(matrix == box_quiver) or np.all(matrix == dreaded_torus_quiver):
        box_or_dreaded_counter += 1
    else:
        not_box_or_dreaded.append(matrix)
assert(box_or_dreaded_counter == 2)

# Removed the box quiver and dreaded torus
print(f'\nRemoving box & dreaded --> NMA\nTotal matrices undecided MA: {len(not_box_or_dreaded)}',flush=True)
print(f'Counts for (MA,NMA): {(len(MA),len(NMA))}\n',flush=True)

#########################################################################################
#%% # Step (6): Generate NMA mutation classes
print('\n# Step (6) #\n',flush=True)
#Create a databse of NMA-mutated quivers (with |em| <= x)
NMA_mutate_set = []
for matrix in NMA:
    NMA_mutate_set += ExchangeGraph(matrix,nma_set_mutate_depth,nma_set_max_edge_multiplcity)
    
#Convert the set to canonical form
NMA_mutate_set = [find_canonical_form(matrix) for matrix in NMA_mutate_set]
NMA_mutate_set = list(np.unique(NMA_mutate_set,axis=0))
    
print(f'Number of mutated-NMA matrices to check against: {len(NMA_mutate_set)}',flush=True)
print(f'Counts for (MA,NMA): {(len(MA),len(NMA))}\n',flush=True)

#########################################################################################
#%% # Step (7): Filter out NMA-mutated quivers
print('\n# Step (7) #\n',flush=True)
NMA_mutated = []
not_NMA_mutated = []
for matrix in not_box_or_dreaded:
    nma_check = False
    for nma_mutate_matrix in NMA_mutate_set:
        if np.max(np.absolute(nma_mutate_matrix)) > 2:
            continue
        if np.all(matrix == nma_mutate_matrix):
            nma_check = True
            break
    if nma_check:
        NMA_mutated.append(matrix)
    else:
        not_NMA_mutated.append(matrix)

NMA += NMA_mutated
        
print(f'Number of undecided which are NMA-mutation-equivalent: {len(NMA_mutated)}\nRemaining undecided: {len(not_NMA_mutated)}',flush=True)
print(f'Counts for (MA,NMA): {(len(MA),len(NMA))}\n',flush=True)

#########################################################################################
#%% # Step (8): Mutate the remaining undecided quivers until acyclic or in NMA_mutate_set
print('\n# Step (8) #\n',flush=True)
print(f'Running checks of {len(not_NMA_mutated)} matrices...',flush=True)
undecided = []
nma_outcome_counters = np.zeros(5)
for matrix_idx, matrix in enumerate(not_NMA_mutated):
    print(f'Matrix {matrix_idx}:',flush=True)
    MA_check, outcome_idx = EG_Search_MAChecker(matrix,final_check_depth,NMA_mutate_set)
    print(f'...outcome: {MA_check} ({outcome_idx})\n',flush=True)
    if MA_check == True:
        MA.append(matrix)
    elif MA_check == False:
        NMA.append(matrix)
        nma_outcome_counters[outcome_idx] += 1
    else: 
        undecided.append(matrix)

print(f'Counts for (MA,NMA): {(len(MA),len(NMA))}',flush=True)
print(f'Number of undecided matrices: {len(undecided)}',flush=True)
print(f'NMA outcome counts: {nma_outcome_counters}',flush=True)

#Save lists to separate files
with open('MA.txt','w') as ma_file:
    for matrix in MA:
        ma_file.write(str(matrix.tolist())+'\n')
with open('NMA.txt','w') as nma_file:
    for matrix in NMA:
        nma_file.write(str(matrix.tolist())+'\n')
with open('undecided.txt','w') as u_file:
    for matrix in undecided:
        u_file.write(str(matrix.tolist())+'\n')
