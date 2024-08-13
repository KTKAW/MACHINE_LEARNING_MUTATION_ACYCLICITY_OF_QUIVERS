import numpy as np
from math import comb
import datetime
import itertools 
import graph_tool as gt 
from graph_tool.all import *
from sage.all import * 

depth_NMA = 5
depth_MA = 7

def flatten_list(nested_list): 
    ''' 
    Converts an nested python list into a flatten 1-D python array.
    '''
    return list(itertools.chain(*nested_list))


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

def quiver_obtainer(q):
    '''
    Input a lists of list of quivers of the form: 
    
    [[quiver,integer],.....,[quiver,integer]]
    
    Returns a list of quivers
    
    '''
    output =[]
    for i in Sequence_Iteration(q):
        #print('CHEESE')
        #print(i)
        output.append(i[0])
    return output    
    


def CLASS_DATA_Generation(intial_quiver, depth): 
    '''Generates mutated quivers from an intial quiver up to some depth
    Input: An intial Quiver from Sage Maths
    Output:A list of mutated Quivers generated from the intial quiver up to depth "depth".
    '''
    
    
    ########################################
    
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
    
    #######################################################################
    print('Output_list_length:',len(output_list))

   
    return output_list
      
  
       
   








def exchangematrix(setting,a=1,b=1,c=1):
    ''' A function which generates the exchange matrix for Rank 4 Quivers.
    There are 4 classes of exchange matrices
    
    '''
    if setting == 'A4': 
        bij = [[0,a,0,0],[-a,0,b,0],[0,-b,0,c],[0,0,-c,0]] 
    
    elif setting == 'D4': 
         bij = [[0,a,0,0],[-a,0,b,c],[0,-b,0,0],[0,-c,0,0]]
        #bij = [[0,a,b,c],[-a,0,0,0],[-b,0,0,0],[-c,0,0,0]]  
        
    elif setting == 'M1':
        bij = [[0,-2,2,a],[2,0,-2,c],[-2,2,0,b],[-a,-c,-b,0]]
            
    
    elif setting == 'M2':
        bij = [[0,-2,2,-a],[2,0,-2,c],[-2,2,0,b],[a,-c,-b,0]]
            
        
    elif setting == 'M3':
        bij = [[0,-2,2,-a],[2,0,-2,c],[-2,2,0,-b],[a,-c,b,0]]
             
    
    elif setting == 'NM1':
        bij = [[0,-3,3,a],[3,0,-3,c],[-3,3,0,b],[-a,-c,-b,0]]
            
    
    elif setting == 'NM2':
        bij = [[0,-3,3,-a],[3,0,-3,c],[-3,3,0,b],[a,-c,-b,0]]
            
        
    elif setting == 'NM3':
        bij = [[0,-3,3,-a],[3,0,-3,c],[-3,3,0,-b],[a,-c,b,0]]
    
    elif setting == 'B1': 
        bij = [[0,2,0,-2],[-2,0,2,0],[0,-2,0,2],[2,0,-2,0]]
        #bij = [[0,a,0,d],[-a,0,d,0],[0,-d,0,-a],[-d,0,a,0]]
        
    elif setting == 'B2': 
        bij = [[0,3,0,-2],[-3,0,2,0],[0,-2,0,3],[2,0,-3,0]]
        #bij = [[0,a,0,d],[-a,0,d,0],[0,-d,0,-a],[-d,0,a,0]]
    
    elif setting == 'B3': 
        bij = [[0,3,0,-3],[-3,0,3,0],[0,-3,0,3],[3,0,-3,0]]
        #bij = [[0,a,0,d],[-a,0,d,0],[0,-d,0,-a],[-d,0,a,0]]
   
 
    else: 
        bij = [[0,a,0,0],[-a,0,b,0],[0,-b,0,c],[0,0,-c,0]]
    
    return matrix(bij)    




#########################################################
########################################################
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



def data_writer_ACYCLIC(data,name): 
    '''
    A function which takes in a list of SAGE_MATH QUIVERS for ACYCLIC Quivers, 
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

#######################
# 
def data_writer_NONACYCLIC(data,name): 
    '''
    A function which takes in a list of SAGE_MATH QUIVERS for NON ACYCLIC Quivers, 
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
    save_name = file_names(types = name)+'_'+'_type_1'+'.txt'    #updates save name for exchange matrix file
    save_name_cat = file_names(types = name)+'_'+'_type_1'+'_'+'cat'+'.txt' #updates save name for cat file
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






#########################################################
###########################################################




#Generating_data for NMA_part_1
nma_part_1 = []
nma_p1_name = ['M1','M2','M3','NM1','NM2','NM3']
for i in nma_p1_name:
    for k in range(0,4):
        for l in range(0,4):
            for m in range(0,4):
                if k == l == m == 0:
                    continue 
                else: 
                    quivs = ClusterSeed(exchangematrix(i,a=k,b=l,c=m)).quiver()
                    nma_part_1.append(quivs)
    


for k in ['B1','B2','B3']:
    nma_part_1.append(ClusterSeed(exchangematrix(k)).quiver())

#mutation 
final_NMA =[CLASS_DATA_Generation(i,depth =depth_NMA) for i in Sequence_Iteration(nma_part_1)]
final_NMA_V2 = flatten_list([quiver_obtainer(i) for i in Sequence_Iteration(final_NMA)])    
data_writer_NONACYCLIC(final_NMA_V2,'NMA_ALL_Depth_{}'.format(depth_NMA))    



#Mutation Acyclic Generation Generation 

mutation_acyclic_name = ['A4','D4']

MA_list = [] 
for name in mutation_acyclic_name:
    for i in range(1,4):
        for j in range(1,4):
            for k in range(1,4): 
                quivs = ClusterSeed(exchangematrix(name,a=i,b=j,c=k)).quiver()
                MA_list.append(quivs)
                
            






#mutation 
final_MA =[CLASS_DATA_Generation(i,depth =depth_MA) for i in Sequence_Iteration(MA_list)] 

final_MA_V2 = flatten_list([quiver_obtainer(i) for i in Sequence_Iteration(final_MA)])


       

data_writer_ACYCLIC(final_MA_V2,'MA_ALL_DEPTH_{}'.format(depth_MA))