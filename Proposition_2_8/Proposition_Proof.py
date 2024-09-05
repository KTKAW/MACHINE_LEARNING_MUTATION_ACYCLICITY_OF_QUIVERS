'''
Script to Prove Proposition 2.8 in the Paper.
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



def CLASS_DATA_Generation(intial_quiver, depth): 
    '''Generates mutated quivers from an intial quiver up to some depth
    Input: An intial Quiver from Sage Maths
    Output:A list of mutated Quivers generated from the intial quiver up to depth "depth".
    '''    
    output_list = []
    for dd in Sequence_Iteration(list(range(0,depth))): # Runs over each depth we wish to consider
       
        if dd == 0: #Checks if we are in the first depth run  
            output_list.append(list((intial_quiver,-1)))
            
            for h in Sequence_Iteration(range(0,4)):#runs a for loop over each vertex to mutate
                
                newquiver = intial_quiver.mutate(h,inplace=False) #Generates a new mutated quiver, from mutation at vertex h
                output_list.append(list((newquiver,h))) #appends mutated quiver to the quiver list for this depth run.         
            beginning_of_run = 1 #sets the position in b_matrix_list that we wish to iterate over from in the next isomorphism check 
            end_of_run = len(output_list) - 1 #set the last position in b_matrix_list that we wish to iterate over in the next isopmorphism
        else: #Instructions for every other depth run
            print('\n\nNEW_DEPTH_{}\n\n'.format(dd+1))
            qs_before = output_list[beginning_of_run:end_of_run+1]
            length = len(qs_before) 
            print("Number of Quivers to mutate:",len(qs_before))
            for position in range(length):#a for loop over the number of quivers from the last depth run
                vertex_list = [0,1,2,3] #vertices in our graph
                vertex_list_copy = vertex_list.copy()#makes a copy of the vertex_list
                term_to_delete = qs_before[position][1]#the vertex that was mutated to generate the quiver in the previous depth iteration
                if term_to_delete==-1:#Checks for the intial quiver in the depth
                    continue #Does not mutate over the intial quiver, skips to the next quiver 
                vertex_list_copy.remove(term_to_delete) #deletes the vertex that was mutated in the previous dpeth iteration
                qs = qs_before[position][0] 
    
              
                for h in Sequence_Iteration(vertex_list_copy):#Runs over mutations over each vertex 'h'
                    newquiver = qs.mutate(h,inplace=False) #mutates the quiver at vertex h to give a new quiver  
                    output_list.append(list((newquiver,h))) #appends mutated quiver to the quiver list for this depth run. 
                                     
            beginning_of_run = end_of_run+1 #sets the position in b_matrix_list that we wish to iterate over from in the next isomorphism check 
            end_of_run = len(output_list) - 1 
                          
                       
        print('Number of Quivers:',len(output_list))
    print('Output_list_length:',len(output_list))
   
    return output_list


#################################################################################################
# NON ACYCLIC QUIVER GENERATION (All Possible NMA Quivers for Our dataset)
################################################################################################

non_acyclic_1a1_data = exchangematrix('NONACYCLIC1A',a=2,b=2,c=2,d=2)
non_acyclic_1a1 = ClusterSeed(non_acyclic_1a1_data).quiver() 


non_acyclic_1a2_data = exchangematrix('NONACYCLIC1B',a=2,b=2,c=2,d=2)
non_acyclic_1a2 = ClusterSeed(non_acyclic_1a2_data).quiver() 


non_acyclic_1a3_data = exchangematrix('NONACYCLIC1C',a=2,b=2,c=2,d=2)
non_acyclic_1a3 = ClusterSeed(non_acyclic_1a3_data).quiver() 


non_acyclic_1b1_data = exchangematrix('NONACYCLIC1A',a=2,b=2,c=2,d=-2)
non_acyclic_1b1= ClusterSeed(non_acyclic_1b1_data).quiver() 


non_acyclic_1b2_data = exchangematrix('NONACYCLIC1B',a=2,b=2,c=2,d=-2)
non_acyclic_1b2 = ClusterSeed(non_acyclic_1b2_data).quiver() 


non_acyclic_1b3_data = exchangematrix('NONACYCLIC1C',a=2,b=2,c=2,d=-2)
non_acyclic_1b3 = ClusterSeed(non_acyclic_1b3_data).quiver() 


non_acyclic_1c1_data = exchangematrix('NONACYCLIC1A',a=2,b=2,c=2,d=1)
non_acyclic_1c1 = ClusterSeed(non_acyclic_1c1_data).quiver() 


non_acyclic_1c2_data = exchangematrix('NONACYCLIC1B',a=2,b=2,c=2,d=1)
non_acyclic_1c2 = ClusterSeed(non_acyclic_1c2_data).quiver() 


non_acyclic_1c3_data = exchangematrix('NONACYCLIC1C',a=2,b=2,c=2,d=1)
non_acyclic_1c3 = ClusterSeed(non_acyclic_1c3_data).quiver() 


non_acyclic_1d1_data = exchangematrix('NONACYCLIC1A',a=2,b=2,c=2,d=-1)
non_acyclic_1d1= ClusterSeed(non_acyclic_1d1_data).quiver() 


non_acyclic_1d2_data = exchangematrix('NONACYCLIC1B',a=2,b=2,c=2,d=-1)
non_acyclic_1d2 = ClusterSeed(non_acyclic_1d2_data).quiver() 


non_acyclic_1d3_data = exchangematrix('NONACYCLIC1C',a=2,b=2,c=2,d=-1)
non_acyclic_1d3 = ClusterSeed(non_acyclic_1d3_data).quiver() 


non_acyclic_2_data = exchangematrix('NONACYCLIC2',a=2,b=2,c=2,d=2)
non_acyclic_2 = ClusterSeed(non_acyclic_2_data).quiver()


non_acyclic_3a1_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=2,e=2,f=2)
non_acyclic_3a1 = ClusterSeed(non_acyclic_3a1_data).quiver()


non_acyclic_3a2_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-2,e=-2,f=-2)
non_acyclic_3a2 = ClusterSeed(non_acyclic_3a2_data).quiver()


non_acyclic_3b1_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=1,e=1,f=1)
non_acyclic_3b1 = ClusterSeed(non_acyclic_3b1_data).quiver()


non_acyclic_3b2_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-1,e=-1,f=-1)
non_acyclic_3b2 = ClusterSeed(non_acyclic_3b2_data).quiver()


non_acyclic_3c1a_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=2,e=1,f=2)
non_acyclic_3c1a= ClusterSeed(non_acyclic_3c1a_data).quiver()


non_acyclic_3c1b_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=1,e=2,f=2)
non_acyclic_3c1b= ClusterSeed(non_acyclic_3c1b_data).quiver()


non_acyclic_3c1c_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=2,e=2,f=1)
non_acyclic_3c1c= ClusterSeed(non_acyclic_3c1c_data).quiver()


non_acyclic_3c2a_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-2,e=-1,f=-2)
non_acyclic_3c2a= ClusterSeed(non_acyclic_3c2a_data).quiver()


non_acyclic_3c2b_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-1,e=-2,f=-2)
non_acyclic_3c2b= ClusterSeed(non_acyclic_3c2b_data).quiver()


non_acyclic_3c2c_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-2,e=-2,f=-1)
non_acyclic_3c2c= ClusterSeed(non_acyclic_3c2c_data).quiver()


non_acyclic_3d1a_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=2,e=1,f=1)
non_acyclic_3d1a = ClusterSeed(non_acyclic_3d1a_data).quiver()


non_acyclic_3d1b_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=1,e=2,f=1)
non_acyclic_3d1b = ClusterSeed(non_acyclic_3d1b_data).quiver()


non_acyclic_3d1c_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=1,e=1,f=2)
non_acyclic_3d1c = ClusterSeed(non_acyclic_3d1c_data).quiver()


non_acyclic_3d2a_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-2,e=-1,f=-1)
non_acyclic_3d2a = ClusterSeed(non_acyclic_3d2a_data).quiver()


non_acyclic_3d2b_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-1,e=-2,f=-1)
non_acyclic_3d2b = ClusterSeed(non_acyclic_3d2b_data).quiver()


non_acyclic_3d2c_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-1,e=-1,f=-2)
non_acyclic_3d2c = ClusterSeed(non_acyclic_3d2c_data).quiver()


non_acyclic_4a1a_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-1,e=-1,f=0)
non_acyclic_4a1a = ClusterSeed(non_acyclic_4a1a_data).quiver()


non_acyclic_4a1b_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-1,e=0,f=-1)
non_acyclic_4a1b = ClusterSeed(non_acyclic_4a1b_data).quiver()


non_acyclic_4a1c_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=0,e=-1,f=-1)
non_acyclic_4a1c = ClusterSeed(non_acyclic_4a1c_data).quiver()


non_acyclic_4a2a_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=1,e=1,f=0)
non_acyclic_4a2a = ClusterSeed(non_acyclic_4a2a_data).quiver()


non_acyclic_4a2b_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=1,e=0,f=1)
non_acyclic_4a2b = ClusterSeed(non_acyclic_4a2b_data).quiver()


non_acyclic_4a2c_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=0,e=1,f=1)
non_acyclic_4a2c = ClusterSeed(non_acyclic_4a2c_data).quiver()


non_acyclic_4a3a_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-2,e=-2,f=0)
non_acyclic_4a3a = ClusterSeed(non_acyclic_4a3a_data).quiver()


non_acyclic_4a3b_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-2,e=0,f=-2)
non_acyclic_4a3b = ClusterSeed(non_acyclic_4a3b_data).quiver()


non_acyclic_4a3c_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=0,e=-2,f=-2)
non_acyclic_4a3c = ClusterSeed(non_acyclic_4a3c_data).quiver()


non_acyclic_4a4a_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=2,e=2,f=0)
non_acyclic_4a4a = ClusterSeed(non_acyclic_4a4a_data).quiver()


non_acyclic_4a4b_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=2,e=0,f=2)
non_acyclic_4a4b = ClusterSeed(non_acyclic_4a4b_data).quiver()


non_acyclic_4a4c_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=0,e=2,f=2)
non_acyclic_4a4c = ClusterSeed(non_acyclic_4a4c_data).quiver()


non_acyclic_4a5a_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-1,e=-2,f=0)
non_acyclic_4a5a = ClusterSeed(non_acyclic_4a5a_data).quiver()


non_acyclic_4a5b_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-1,e=0,f=-2)
non_acyclic_4a5b = ClusterSeed(non_acyclic_4a5b_data).quiver()


non_acyclic_4a5c_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=0,e=-1,f=-2)
non_acyclic_4a5c = ClusterSeed(non_acyclic_4a5c_data).quiver()


non_acyclic_4a6a_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=1,e=2,f=0)
non_acyclic_4a6a = ClusterSeed(non_acyclic_4a6a_data).quiver()


non_acyclic_4a6b_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=1,e=0,f=2)
non_acyclic_4a6b = ClusterSeed(non_acyclic_4a6b_data).quiver()


non_acyclic_4a6c_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=0,e=1,f=2)
non_acyclic_4a6c = ClusterSeed(non_acyclic_4a6c_data).quiver()


non_acyclic_4a7a_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-2,e=-1,f=0)
non_acyclic_4a7a = ClusterSeed(non_acyclic_4a7a_data).quiver()


non_acyclic_4a7b_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=-2,e=0,f=-1)
non_acyclic_4a7b = ClusterSeed(non_acyclic_4a7b_data).quiver()


non_acyclic_4a7c_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=0,e=-2,f=-1)
non_acyclic_4a7c = ClusterSeed(non_acyclic_4a7c_data).quiver()


non_acyclic_4a8a_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=2,e=1,f=0)
non_acyclic_4a8a = ClusterSeed(non_acyclic_4a8a_data).quiver()


non_acyclic_4a8b_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=2,e=0,f=1)
non_acyclic_4a8b = ClusterSeed(non_acyclic_4a8b_data).quiver()


non_acyclic_4a8c_data = exchangematrix('NONACYCLIC3',a=2,b=2,c=2,d=0,e=2,f=1)
non_acyclic_4a8c = ClusterSeed(non_acyclic_4a8c_data).quiver()

#Dread Torus Inclusion: 
non_acyclic_dreaded_data=  matrix([[0,1,1,-1],[-1,0,-1,2],[-1,1,0,-1],[1,-2,1,0]])
non_acyclic_dreaded = ClusterSeed(non_acyclic_dreaded_data).quiver()

# Concatenate the lists
non_acyclic_list = [non_acyclic_1a1,non_acyclic_1a2,non_acyclic_1a3, non_acyclic_1b1,non_acyclic_1b2,non_acyclic_1b3,non_acyclic_1c1,non_acyclic_1c2,non_acyclic_1c3,non_acyclic_1d1,non_acyclic_1d2,non_acyclic_1d3,non_acyclic_2,non_acyclic_3a1,non_acyclic_3a2,non_acyclic_3b1,non_acyclic_3b2,non_acyclic_3c1a,non_acyclic_3c1b,non_acyclic_3c1c,non_acyclic_3c2a,non_acyclic_3c2b,non_acyclic_3c2c,non_acyclic_3d1a,non_acyclic_3d1b,non_acyclic_3d1c,non_acyclic_3d2a,non_acyclic_3d2b,non_acyclic_3d2c,non_acyclic_4a1a,
                   non_acyclic_4a1b,non_acyclic_4a1c,non_acyclic_4a2a,
                   non_acyclic_4a2b,non_acyclic_4a2c,non_acyclic_4a3a,
                   non_acyclic_4a3b,non_acyclic_4a3c,non_acyclic_4a4a,
                   non_acyclic_4a4b,non_acyclic_4a4c,non_acyclic_4a5a,
                   non_acyclic_4a5b,non_acyclic_4a5c,non_acyclic_4a6a,
                   non_acyclic_4a6b,non_acyclic_4a6c,non_acyclic_4a7a,
                   non_acyclic_4a7b,non_acyclic_4a7c,non_acyclic_4a8a,
                   non_acyclic_4a8b,non_acyclic_4a8c,non_acyclic_dreaded]

#Extending non_acyclic_list 
temp_list = [] #Define Temporary List
for i in non_acyclic_list:
    new_quivers = CLASS_DATA_Generation(i,depth=4)
    for k in new_quivers:
        temp_list.append(k[0])

non_acyclic_list = temp_list.copy()
temp_list.clear()





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





# PROPOSITION PROOF

#Loads All Isomorphism Classes
with open('mutation_class_list.pkl', 'rb') as f:
    mutation_class_list = pickle.load(f)
print('\n')
print(f'Number of Isomorphism Classes: {len(mutation_class_list)}') # Prints Number of Isomorphism Classes: 665
print('\n')
#Removes all NMA Quivers from mutation_class_list
iso_killer = [edge_list_creator(x) for x in Sequence_Iteration(non_acyclic_list)] #Converts NMA quivers to graph_tool
get_rid = iso_finder(mutation_class_list,iso_killer) #Finds all the NMA quivers in the Isomorphism class list 

#Saving NMA Class Examples
for i in get_rid:
    image = mutation_class_list[i]
    image.save_image(f'NON_Mutation_Acyclic_Isomorphism_Class_Picture_List_Position_{i}.png')

print('\n')
print(f'Number of Non-Mutation Acyclic Isomorphism Classes: {len(get_rid)}') #Number of Isomorphism Classes
print('\n')
MA = np.delete(mutation_class_list,get_rid) #Deletes NMA quivers, to just give MA Isomorphism Class Quivers

#Saves MA Isomorphism Quivers to a Pickle File
with open('MA.pkl','wb') as ff:
    pickle.dump(MA,ff)


MA_LIST_OUT = MA.tolist() #Converts array of MA quivers to a list
print('\n')
print(f'Number of Mutation Acyclic Isomorphism Classes: {len(MA)}') #Prints Number of Mutation Acyclic Quiver Classes: 644
print('\n')

#Saves Image of MA Quivers
for i in range(len(MA)):
    image = MA[i]
    image.save_image(f'Mutation_Acyclic_Isomorphism_Class_Picture_{i+1}.png') 


#Proving Quivers are Acyclic 

def Acyclicity_Test(quiver): 
    '''A function which determines whether a SAGEMATH Quiver is Acyclic
       INPUT: SAGEMATH Quiver
       OUTPUT: Boolean
    '''
    quiver_b_matrix = quiver.b_matrix()
    quiver_network = nx.from_numpy_array(np.array(quiver_b_matrix),parallel_edges=True,create_using=nx.MultiDiGraph()).copy()
    is_Acyclic= nx.is_directed_acyclic_graph(quiver_network) #Tests if a graph is connected or not
    return is_Acyclic

#Beginning of Acyclicty Proof: 

counter = 0
Acyclic_Quivers = []
for i in MA:
    counter+=1
    print(f'This is Quiver {counter}/{len(MA)}')
    a_test=False 
    list_of_Quivers = CLASS_DATA_Generation(i,depth=5)
    for k in range(len(list_of_Quivers)):
        print(f'This is Quiver {k+1} out of {len(list_of_Quivers)}')
        quiver_to_test = list_of_Quivers[k][0]
        a_test = Acyclicity_Test(quiver_to_test)
        if a_test == True:
            Acyclic_Quivers.append(counter-1)   
            break
        else:
            continue         

with open('Acyclic_Quivers.txt','w+') as p:
    p.write(f'Number of Acyclic Quivers: {len(Acyclic_Quivers)}/{len(MA)}\n')
    p.write(str(Acyclic_Quivers))
p.close() 

Bad_Quivers = np.delete(MA,Acyclic_Quivers).tolist()

#Saving Diffcult to Prove Quivers

with open('Diffcult_To_Prove_Quivers.pkl','wb') as fff:
    pickle.dump(Bad_Quivers,fff)

print('---'*5,'\tProgram Finished\t','---'*5)    