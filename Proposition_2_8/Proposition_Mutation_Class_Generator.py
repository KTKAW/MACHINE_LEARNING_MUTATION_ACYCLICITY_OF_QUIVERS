'''
Script to Generate the required datasets to prove proposition 2.8 in the Paper.
'''
# Import libraries
import numpy as np
from sage.all import *
from math import comb
import networkx as nx 
import datetime
import itertools 
import pickle
import graph_tool as gt 
from graph_tool.all import *

# Define appropriate functions
class Sequence_Iteration:
    ''' 
    A class which iterates over a list 
    '''
    def __init__(self,sequence_list):
        self._sequence_list = sequence_list #Makes the input list a class variable
        self._counter =  0 

    def __iter__(self): # Makes the class an iterator
        return self    

    def __next__(self): #Defines the Rules the iterator must follow
        if self._counter < len(self._sequence_list): 
            list_attr = self._sequence_list[self._counter] 
            self._counter += 1 
            return list_attr 
        else: 
            raise StopIteration  


def exchangematrix(setting,a=1,b=1,c=1,d=1,e=1,f=1):
    ''' A function which generates the exchange matrix for Rank 4 Quivers.
    There are 4 classes of exchange matrices
    
    '''
    if setting == 'ACYCLIC1': 
        bij = [[0,a,0,0],[-a,0,b,0],[0,-b,0,c],[0,0,-c,0]] 
    
    elif setting == 'ACYCLIC2': 
        bij = [[0,a,b,c],[-a,0,0,0],[-b,0,0,0],[-c,0,0,0]]  
    
    elif setting == 'NONACYCLIC1A':
        bij = [[0,a,-c,0],[-a,0,b,0],[c,-b,0,d],[0,0,-d,0]]
        
    elif setting == 'NONACYCLIC1B':
        bij = [[0,a,-c,d],[-a,0,b,0],[c,-b,0,0],[-d,0,0,0]]  
    
    elif setting == 'NONACYCLIC1C':
        bij = [[0,a,-c,0],[-a,0,b,d],[c,-b,0,0],[0,-d,0,0]]
        
    elif setting == 'NONACYCLIC2': 
        bij = [[0,a,0,-d],[-a,0,d,0],[0,-d,0,a],[d,0,-a,0]]
        #bij = [[0,a,0,d],[-a,0,d,0],[0,-d,0,-a],[-d,0,a,0]]
    elif setting == 'NONACYCLIC3': 
        bij = [[0,a,-c,f],[-a,0,b,e],[c,-b,0,d],[-f,-e,-d,0]]
    
    else: 
        bij = [[0,a,0,0],[-a,0,b,0],[0,-b,0,c],[0,0,-c,0]]
    
    return matrix(bij)    


def edge_list_creator(quiver): 
    '''
       Creates a (from a sage math quiver) Graph Tool Graph
       INPUT: Sage Math Quiver
       OUTPUT: Graph Tool Directed Graph
    '''
    edge_list_input = quiver.digraph().edges() #A list of connections between each vertex, 3-tuple => (start_node, end_node,edges between)
    edge_list_output = []
    g = gt.Graph(directed =True) # creates a directed graph
    length = len(edge_list_input) #length of edge_list_input
    for l in range(0,length): #for loop over the length of edge_list_input (integer values)
        tups = edge_list_input[l] #Unpacks tuple from chosen edge list input element
        start, end, amount  = tups #unpacks tuple
        positive, negative = amount#unpacks the positive and negative edge weights from the amount tuples
        if amount == 1: #if there is only one edge between vertices
            edge_list_output.append((start,end)) #appends edge information
        else: #if there are more than one edge between vertices
            for h in range(positive): #iterates over the number of edges between the two vertices 
                edge_list_output.append((start,end)) #appends edge information
    g.add_edge_list(edge_list_output)  #creates a new graph tool graph from the edge_list_output     
    return g.copy() #returns a copied version of the list  


def rank4genm(a=1,b=1,c=1,d=1,e=1,f=1):
    ''' A function which generates the exchange matrix for Rank 4 Quivers as an numpy array.
    There are 4 classes of exchange matrices
    
    '''
    
    bij = [[0,a,b,c],[-a,0,d,e],[-b,-d,0,f],[-c,-e,-f,0]] 
    
    
    return matrix(bij)    


def array_converter(matrix1): 
    '''
    A function which converts a Sage math matrix into a flattened 1-D python list.
    INPUT: SAGE MATH MATRIX
    OUTPUT: 1-D PYTHON LIST
    '''
    array = matrix1.numpy()
    arraynewp1 = array.flatten() 
    arraynew = arraynewp1.astype('float64')   
    return arraynew.tolist()  


def liststripper(lists):
    '''Removes the square brackets from flatten 1-D arrays which have already been converted to lists.
       INPUT: 1-D PYTHON ARRAY
       OUTPUT: STRING
    '''
    list_string = str(lists) #Converts list to string
    list_string_v2 = list_string.replace('[','') 
    list_string_v3 = list_string_v2.replace(']','')
    return list_string_v3


def file_names(types='A4_'):
    '''
    A file name is produced using the current date and time.

    INPUT: STRING - The file name wanted for the output file.
 
    Returns:
    - str:
        The generated file name in the format 'YYYY-MM-DD_HH-MM-SS'.
        This is then added to the string "types"+"_". 
        The output is then given by the string: "types"+"_"+"YYYY_MM_DD_HH_MM_SS"
    
    Example: 
    
    INPUT: "A4"
    SYSTEM_TIME = "2024-07_00-01-00"    
    OUTPUT = "A4_2024_07_00_01_00"
    ''' 
    # Get the current date and time from operating system
    current_datetime = datetime.datetime.now()
 
    # Format the date and time as a string in the desired format
    file_name_part_1 = types+'_'+current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    file_name = file_name_part_1 
 
    return file_name 


def data_writer(data,name): 
    '''
    Converts a list of Sage Math Matrices into a python list, which is then written to a .txt file
    INPUT: data;1-D Python List with entries of Sage Math Matrix.,
           name; File name for txt. file we write the python lists to.
    '''
    save_name = file_names(types = name)+'.txt' 
    save_file = open(save_name,'w')  
    for j in data: 
        toprint = array_converter(j) 
        string_output = liststripper(toprint)
        save_file.write(string_output+'\n')    
    save_file.close() 


def cluster_generator(max_legs = 2):
    '''
    Generates all possible connected Rank 4 Quivers with maximum number of edge lengths "max_legs".
    INPUT: Integer
    OUTPUT: python list of Sage Math Quivers 
    '''
    b_matrix_list = []
    for i in np.arange(-max_legs,max_legs+1):
        for j in np.arange(-max_legs,max_legs+1):
            for k in np.arange(-max_legs,max_legs+1):
                for l in np.arange(-max_legs,max_legs+1):
                    for g in np.arange(-max_legs,max_legs+1):
                        for h in np.arange(-max_legs,max_legs+1): 
                            bij = rank4genm(a=i,b=j,c=k,d=l,e=g,f=h) 
                            quiver_network = ClusterSeed(bij).quiver()
                            quiver_network_2 = nx.from_numpy_array(np.array(bij),parallel_edges=True,create_using=nx.MultiDiGraph()).copy()
                            connection_test= nx.is_connected(quiver_network_2.to_undirected()) #Tests if a graph is connected or not
                            if connection_test == False: 
                                continue
                            else:     
                                b_matrix_list.append(quiver_network)
                            
    print('END')
    return b_matrix_list                       


def array_exchange_matrices_sage(input_list): 
    ''' Generates from a list of networks from networkx, converts them back into numpy arrays (in the networkx format) and reformats them into the format which Sagemaths reads them
    INPUT: List of Numpy (2D) arrays of shape (4,4)
    OUTPUT: List of Numpy (2D) arrays of shape(4,4)
    '''
    output_list = []
    matrix_elements = list(itertools.combinations(range(0,4),2)) #list the positions of echange matrix in the upper right triangle 
    
    
    for sample_list in input_list: 
        bij_matrix_element =[]
        matrix_input = nx.to_numpy_array(sample_list)
        for tups in matrix_elements: 
            i,j =tups 
            
            upper_value = matrix_input[i,j]
            lower_value = matrix_input[j,i]
           
            if upper_value == lower_value: 
                bij_matrix_element.append(0)
                continue
            elif (upper_value== 0) and (lower_value > 0) : 
                bij_matrix_element.append(-lower_value)
                continue
            elif (upper_value > 0 ) and (lower_value == 0):
                bij_matrix_element.append(upper_value)
                continue    
        bij_tuple   = tuple(bij_matrix_element)  
        a1,a2,a3,a4,a5,a6 = bij_tuple 
        bij = rank4genm(a=a1,b=a2,c=a3,d=a4,e=a5,f=a6)
        output_list.append(bij)
    print('Finished')    
    return output_list    

#ISO_KILLER CODE 

def iso_finder_alternative(quivers,list_to_compare): 
    '''
    INPUT 1: LIST OF SAGE MATH QUIVERS 
    INPUT 2: LIST OF SAGE MATH QUIVERS THAT WE USE TO FIND ISOMORHPISMS IN LIST 1
    OUTPUT: The postion of the graphs in INPUT 1, that are ismomorphic to those in INPUT 2
    '''
    gt_quiver_list = [edge_list_creator(x) for x in Sequence_Iteration(quivers)] 
    list_to_compare_gt = [edge_list_creator(x) for x in Sequence_Iteration(list_to_compare)]
    counter = -1
    get_rid_list =[]
    for k in Sequence_Iteration(gt_quiver_list): 
        counter +=1
        for j in Sequence_Iteration(list_to_compare_gt):
            decide = gt.topology.isomorphism(k,j)
            if decide == True: 
                get_rid_list.append(counter)
                break
            else:
                continue 
    return get_rid_list 


def iso_finder(quivers,list_to_compare): 
    '''
    INPUT 1: LIST OF SAGE MATH QUIVERS 
    INPUT 2: LIST OF SAGE MATH QUIVERS THAT WE USE TO FIND ISOMORHPISMS IN LIST 1
    OUTPUT: The postion of the graphs in INPUT 1, that are ismomorphic to those in INPUT 2
    '''
    gt_quiver_list = [edge_list_creator(x) for x in Sequence_Iteration(quivers)] 
    counter = -1
    get_rid_list =[]
    for k in Sequence_Iteration(gt_quiver_list): 
        counter +=1
        for j in Sequence_Iteration(list_to_compare):
            decide = gt.topology.isomorphism(k,j)
            if decide == True: 
                get_rid_list.append(counter)
                break
            else:
                continue 
    return get_rid_list     

#######################################################################################
# Generate the dataset
lists = cluster_generator(max_legs = 2) #Generates all graphs 
print(f'Length of Lists: {len(lists)}')
print('-'*20,'\tDATA GENERATION DONE\t','-'*20)


#Reduce ALL_DATA down to it's isomorphism classes: 
lists2 = np.array(lists.copy()) #makes a copy of list of all graphs and makes it an array
mutation_class_list = []#list where we store representatives of isomorphism graphs
while true:
    if len(lists2) == 0: #exits while loop if lists2 has no elements in it
        break 
    else: 
        term_to_test = lists2[0] #selects the first quiver in lists2
        mutation_class_list.append(term_to_test)#adds it to isomorphism class
        print(f'Number of Items to iterate over {len(lists2)}')
        isomorhpism_removing = iso_finder_alternative(lists2,[term_to_test])#Finds position of quivers which are isomorphic to "term_to_test"
        print(isomorhpism_removing)
        if len(isomorhpism_removing) == 0:
            continue #if no graphs are isomorphic to "term_to_test", restart the while loop.
        else:
            lists2 = np.delete(lists2,isomorhpism_removing) #Deletes all quivers isomorphic to "term_to_test" (including "term_to_test"!) from lists2

print('-----'*20)
print('\n'*20,f'Number of Isomorphism Classes {len(mutation_class_list)}')

#Use Pickle to save mutation_class_list to an external pickle file
with open('mutation_class_list.pkl','wb') as f:
    pickle.dump(mutation_class_list,f)




