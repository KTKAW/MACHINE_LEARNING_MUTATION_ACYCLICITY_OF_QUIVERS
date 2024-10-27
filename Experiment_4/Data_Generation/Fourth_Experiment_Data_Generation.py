'''
Script to exhaustively generate rank 2 quivers with edge multiplicity < 3.
'''
# Import libraries
import numpy as np
from sage.all import *
import datetime
import itertools 


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
    Writes a 1-D Python list to a .txt file
    INPUT: data;1-D Python List with entries of Sage Math Matrix.,
        name; File name for txt. file we write the python lists to.
    '''
    save_name = file_names(types = name)+'.txt' 
    save_file = open(save_name,'w')  
    for j in data: 
        string_output = listtostring(j)
        save_file.write(string_output+'\n')    
    save_file.close() 





def liststripper(lists):
        '''Removes the square brackets from flatten 1-D arrays which have already been converted to lists.
        INPUT: 1-D PYTHON ARRAY
        OUTPUT: STRING
        '''
        list_string = str(lists) #Converts list to string
        list_string_v2 = list_string.replace('[','') 
        list_string_v3 = list_string_v2.replace(']','')
        list_string_v4 = list_string_v3.replace('\n','')
        return list_string_v4.split(',')

def listtostring(lists):
        '''Removes the square brackets from flatten 1-D list 
        INPUT: 1-D PYTHON ARRAY
        OUTPUT: STRING
        '''
        new_string = ''
        for i in range(len(lists)):
            if i == 0: 
                new_string += str(lists[i])
            else: 
                new_string += ',' + str(lists[i])
        return new_string



def string_list_to_int_list(list_entry):
    '''Converts a 1-D Python list of strings to a 1-D list of integers''' 
    return_list = [int(float(x)) for x in list_entry]
    return return_list 

def components_finder_6(lists):
    '''
    INPUT: list of lists of 1 X 16 entries
    OUTPUT: lists of lists of 1 X 6 entries
    '''
    output =[]
    m = Sequence_Iteration(lists)
    for i in m: 
        outs = [i[1],i[2],i[3],i[6],i[7],i[11]]
        output.append(outs)
    return output    

def rank4genm(a=1,b=1,c=1,d=1,e=1,f=1):
    ''' A function which generates the exchange matrix for Rank 4 Quivers as an nested python list.
    There are 4 classes of exchange matrices
    
    '''
    
    bij = [[0,a,b,c],[-a,0,d,e],[-b,-d,0,f],[-c,-e,-f,0]] 
    
    return bij


def numpy_array_copier(array,permutation):
    '''Rearranges the rows and columns of a numpy array based on a permutation of it's rows and columns.
       INPUT:
            - A 4*4 Numpy Array
            - A 4-component python list with the permutations of rows/columns from 
            the original [0,1,2,3] ordering to [a,b,c,d]; where a,b,c,d are ONE of 0,1,2,3 each. 

       OUTPUT: 4*4 Numpy Array whoses componenets have been permuted 
    
    ''' 
    copied_array = np.ones(array.shape) 

    string_indices = ['0','1','2','3']

    copying_dictionary = dict()
    for i in range(len(permutation)): 
        copying_dictionary[string_indices[i]] = permutation[i]

    for i in string_indices:
        for j in string_indices: 
            copied_array[int(i)][int(j)] = array[copying_dictionary[i]][copying_dictionary[j]]    

    return copied_array 


##############################################################################
#Data Generation from Classes Produced by Proof 

MA_Classes = open('MA_classes.txt','r')
MA_Classes_list = components_finder_6([string_list_to_int_list(liststripper(x)) for  x in MA_Classes.readlines()])

NMA_Classes = open('NMA_classes.txt','r')
NMA_Classes_list = components_finder_6([string_list_to_int_list(liststripper(x)) for x in NMA_Classes.readlines()])



MA_Classes_list_v2 = []
NMA_Classes_list_v2 = []


for t in MA_Classes_list:  
    a,b,c,d,e,f = tuple(t)
    MA_Classes_list_v2.append(np.array(rank4genm(a=a,b=b,c=c,d=d,e=e,f=f)))

for t in NMA_Classes_list:
    a,b,c,d,e,f = tuple(t)
    NMA_Classes_list_v2.append(np.array(rank4genm(a=a,b=b,c=c,d=d,e=e,f=f)))


MA_Classes_list.clear()
NMA_Classes_list.clear()

MA_ALL_PERMUTATIONS = []

for i in MA_Classes_list_v2:

    permutation_iterator = itertools.permutations([0,1,2,3])
    for k in permutation_iterator: 
        l = numpy_array_copier(i,k)
        MA_ALL_PERMUTATIONS.append(l)

NMA_ALL_PERMUTATIONS = []

for i in NMA_Classes_list_v2:

    permutation_iterator = itertools.permutations([0,1,2,3])
    for k in permutation_iterator: 
        l = numpy_array_copier(i,k)
        NMA_ALL_PERMUTATIONS.append(l)

MA_ALL_PERMUTATIONS = [x.flatten().tolist() for x in MA_ALL_PERMUTATIONS]
NMA_ALL_PERMUTATIONS = [x.flatten().tolist() for x in NMA_ALL_PERMUTATIONS]


MA_SET = set()
for i in MA_ALL_PERMUTATIONS: 
    MA_SET.add(tuple(i))

NMA_SET = set()
for i in NMA_ALL_PERMUTATIONS:
    NMA_SET.add(tuple(i))

print(f'NUMBER OF MA GRAPHS: {len(MA_SET)}')
print(f'NUMBER OF NMA GRAPHS {len(NMA_SET)}')    

print(f'TOTAL NUMBER OF GRAPHS: {len(MA_SET)+len(NMA_SET)}')

MA_FINAL = [list(x) for x in MA_SET]
NMA_FINAL = [list(x) for x in NMA_SET] 

data_writer(NMA_FINAL,'Fourth_Experiment_NMA_data')#Saves NMA quivers to a .txt file
data_writer(MA_FINAL,'Fourth_Experiment_MA_data')#Saves data set with all the MA quivers to a .txt file

