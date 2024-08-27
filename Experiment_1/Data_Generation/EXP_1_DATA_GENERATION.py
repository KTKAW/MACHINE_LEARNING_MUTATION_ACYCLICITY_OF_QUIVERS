'''
Script to generate mutation classes of quivers, saving their outputs as adjacency matrices in separate txt files.
'''
import numpy as np
import datetime
from sage.all import * 


class Sequence_Iteration:
    ''' 
    A class which iterates over a list 
    '''
    def __init__(self,sequence_list):
        self._sequence_list = sequence_list #Makes the input list a class variable
        self._counter =  0 #Defines a counter variable for each instance. Start from 0.

    def __iter__(self): # Makes the class an iterator
        return self    

    def __next__(self): #Defines the Rules the iterator must follow
        if self._counter < len(self._sequence_list): # Conditional Statement: If the counter variable is smaller than the list 'sequence_list'.
            list_attr = self._sequence_list[self._counter] 
            self._counter += 1 # Increases counter variable counter variable by 1. 
            return list_attr #Returns the assigned value of sequence_list[self._counter].
        else: 
            raise StopIteration  #Stops the iteration if we reach the end of the list


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
    """
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
    """
 
    # Get the current date and time from operating system
    current_datetime = datetime.datetime.now()
 
    # Format the date and time as a string in the desired format
    file_name_part_1 = types+'_'+current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    file_name = file_name_part_1 
 
    return file_name


def data_writer_A4(data,name): 
    '''
    A function which takes in a list of SAGE_MATH QUIVERS for A4 Quivers, 
    and converts each Quiver in that list to its exchange matrix (also known as a b_matrix), 
    converts it to list, and then onto a string. 

    For each quiver, a string corresponding to its exchange matrix is saved to a .txt file. 

    INPUT: 1) A 1-D python list filled with 
           2) The name we wish to include 
    
    '''
    counter = 0 #A counting variable to keep track of the number of quivers. 
    matrixlist = [] #An empty list which we will append sage math Quivers b_matrix() to.
    for quiv in data: #A 'for' loop interating over entries in the sage math quiver 
        matrixprint = quiv.b_matrix() #we convert each quiver into its exchange matrix (SAGE MATH MATRIX)
        matrixlist.append(matrixprint) #Append each exchange matrix to the list matrixlist
    save_name = file_names(types = name)+'_type_0'+'.txt' #updates save name for exchange matrix file
    save_name_cat = file_names(types =name)+'_type_0'+'_'+'cat'+'.txt'#updates save name for cat file
    save_file = open(save_name,'w')
    save_file_cat = open(save_name_cat,'w')    
    for j in matrixlist: 
        toprint = array_converter(j) 
        string_output = liststripper(toprint)
        save_file.write(string_output+'\n')
        if counter == 0: 
            save_file_cat.write(str(0))
        else:
            save_file_cat.write('\n')
            save_file_cat.write(str(0))
        counter += 1     
    save_file.close() 
    save_file_cat.close()


def data_writer_D4(data,name): 
    '''
    A function which takes in a list of SAGE_MATH QUIVERS for D4 Quivers, 
    and converts each Quiver in that list to its exchange matrix (also known as a b_matrix), 
    converts it to list, and then onto a string. 

    For each quiver, a string corresponding to its exchange matrix is saved to a .txt file. 

    INPUT: 1) A 1-D python list filled with 
           2) The name we wish to include 
    
    '''
    counter = 0
    matrixlist = []
    for quiv in data: 
        matrixprint = quiv.b_matrix() 
        matrixlist.append(matrixprint)
    save_name = file_names(types = name)+'_type_1'+'.txt' 
    save_name_cat = file_names(types =name)+'_type_1'+'_'+'cat'+'.txt'
    save_file = open(save_name,'w')
    save_file_cat = open(save_name_cat,'w')    
    for j in matrixlist: 
        toprint = array_converter(j) 
        string_output = liststripper(toprint)
        save_file.write(string_output+'\n')
        if counter == 0: 
            save_file_cat.write(str(1))
        else:
            save_file_cat.write('\n')
            save_file_cat.write(str(1))
        counter += 1     
    save_file.close() 
    save_file_cat.close()


def data_writer_NONACYCLIC_1(data,name): 
    '''
    A function which takes in a list of SAGE_MATH QUIVERS for NMA1 Quivers, 
    and converts each Quiver in that list to its exchange matrix (also known as a b_matrix), 
    converts it to list, and then onto a string. 

    For each quiver, a string corresponding to its exchange matrix is saved to a .txt file. 

    INPUT: 1) A 1-D python list filled with 
           2) The name we wish to include 
    
    '''
    counter = 0
    matrixlist = []
    for quiv in data: 
        matrixprint = quiv.b_matrix() 
        matrixlist.append(matrixprint)
    save_name = file_names(types = name)+'_'+'_type_2'+'.txt'   
    save_name_cat = file_names(types = name)+'_'+'_type_2'+'_'+'cat'+'.txt'
    save_file = open(save_name,'w')
    save_file_cat = open(save_name_cat,'w')
    for j in matrixlist: 
        toprint = array_converter(j)
        string_output=liststripper(toprint)
        save_file.write(string_output+'\n')
        if counter == 0: 
            save_file_cat.write(str(2))
        else: 
            save_file_cat.write('\n')
            save_file_cat.write(str(2))
        counter += 1   
    save_file.close()
    save_file_cat.close()


def data_writer_NONACYCLIC_2(data,name): 
    '''
    A function which takes in a list of SAGE_MATH QUIVERS for NMA2 Quivers, 
    and converts each Quiver in that list to its exchange matrix (also known as a b_matrix), 
    converts it to list, and then onto a string. 

    For each quiver, a string corresponding to its exchange matrix is saved to a .txt file. 

    INPUT: 1) A 1-D python list filled with 
           2) The name we wish to include 
    
    '''
    counter = 0
    matrixlist = []
    for quiv in data: 
        matrixprint = quiv.b_matrix() 
        matrixlist.append(matrixprint)
    save_name = file_names(types = name)+'_'+'_type_3'+'.txt'   
    save_name_cat = file_names(types = name)+'_'+'_type_3'+'_'+'cat'+'.txt'
    save_file = open(save_name,'w')
    save_file_cat = open(save_name_cat,'w')
    for j in matrixlist: 
        toprint = array_converter(j)
        string_output=liststripper(toprint)
        save_file.write(string_output+'\n')
        if counter == 0: 
            save_file_cat.write(str(3))
        else: 
            save_file_cat.write('\n')
            save_file_cat.write(str(3))
        counter += 1   
    save_file.close()
    save_file_cat.close()


def exchangematrix(setting,a=1,b=1,c=1,d=1):
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
    else: 
        bij = [[0,a,0,0],[-a,0,b,0],[0,-b,0,c],[0,0,-c,0]]
    
    return matrix(bij)   

#########################################################################################
# Data Generation
# Define the starting quivers

#A4 Quiver Starting Graph type 1

a4_graph_data = exchangematrix('ACYCLIC1',a=1,b=1,c=1) #Generates the exchange matrix quiver in terms of a sage math matrix
a4_graph = ClusterSeed(a4_graph_data).quiver()

#A4 Quiver Starting Graph type 2

a4_graph_data_2 = exchangematrix('ACYCLIC1',a=2,b=2,c=2)#Generates the exchange matrix quiver in terms of a sage math matrix
a4_graph_2 = ClusterSeed(a4_graph_data_2).quiver()

#A4 Quiver Starting Graph type 3

a4_graph_data_3 = exchangematrix('ACYCLIC1',a=1,b=-2,c=1)#Generates the exchange matrix quiver in terms of a sage math matrix
a4_graph_3 = ClusterSeed(a4_graph_data_3).quiver()

#D4 Quiver Starting Graph type 1

d4_graph_data = exchangematrix('ACYCLIC2',a=1,b=1,c=1)#Generates the exchange matrix quiver in terms of a sage math matrix
d4_graph = ClusterSeed(d4_graph_data).quiver()

#D4 Quiver Starting Graph type 2

d4_graph_data_2 = exchangematrix('ACYCLIC2',a=1,b=1,c=2)#Generates the exchange matrix quiver in terms of a sage math matrix
d4_graph_2 = ClusterSeed(d4_graph_data_2).quiver()

#Non Acyclic 1 Starting Graph

non_acyclic_1_data = exchangematrix('NONACYCLIC1A',a=2,b=2,c=2,d=2)#Generates the exchange matrix quiver in terms of a sage math matrix
non_acyclic_1 = ClusterSeed(non_acyclic_1_data).quiver()

#Non Acyclic 2 Starting Graph

non_acyclic_2_data = exchangematrix('NONACYCLIC2',a=2,b=2,c=2,d=2)#Generates the exchange matrix quiver in terms of a sage math matrix
non_acyclic_2 = ClusterSeed(non_acyclic_2_data).quiver()



#########################################################################################
# Generate the mutation classes

depth_max = 8 #DATA GENERATION DEPTH

#A4 Quiver Data Generation
print('A4 Quiver Data Generation')
a4_data_1 = CLASS_DATA_Generation(a4_graph,depth=depth_max)
print('\nA4_2\n')
a4_data_2 = CLASS_DATA_Generation(a4_graph_2,depth=depth_max) 
print(len(a4_data_2))
print('\n A4_3 \n')
a4_data_3 = CLASS_DATA_Generation(a4_graph_3,depth = depth_max)
print(len(a4_data_3))

a4_data = a4_data_1+a4_data_2+a4_data_3
a4_data_a = np.array(a4_data)
a4_data = a4_data_a[:,0].flatten()
print('A4_LENGTH', len(a4_data))

data_writer_A4(a4_data, name='A4_Extended_FULL')


#D4 Quiver Data Generation: 
print('D4 Quiver Data Generation')
print('D4 1')
d4_data_1 = CLASS_DATA_Generation(d4_graph,depth=depth_max)
print('D4 2')
d4_data_2 = CLASS_DATA_Generation(d4_graph_2,depth=depth_max)
d4_data = np.array(d4_data_1+d4_data_2)
d4_data_final = d4_data[:,0].flatten()

data_writer_D4(d4_data_final, name='D4_Extended_FULL')


#NONACYCLIC1 DATA GENERATION
print('Non-Acyclic 1 Data Generation')
nonacyclic1_data = np.array(CLASS_DATA_Generation(non_acyclic_1,depth = depth_max))[:,0].flatten()

data_writer_NONACYCLIC_1(nonacyclic1_data,name='NONACYCLIC1_Extended_FULL')


#NONACYCLIC2 DATA GENERATION
print('NON-ACYLIC2_Data_Generation')
nonacyclic2_data =np.array(CLASS_DATA_Generation(non_acyclic_2, depth=depth_max))[:,0].flatten()

data_writer_NONACYCLIC_2(nonacyclic2_data,name='NONACYCLIC2_Extended_FULL')


print('END OF PROGRAM')

