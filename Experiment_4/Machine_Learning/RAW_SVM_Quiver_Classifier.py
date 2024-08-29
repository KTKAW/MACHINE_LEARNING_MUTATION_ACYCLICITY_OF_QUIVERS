'''
Script to learn mutation-acyclicity of quivers (using their adjacency upper triangular representation), via support vector machines.
'''
# Import libraries
import numpy as np
from sage.all import * 
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef

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


def data_reading_exchange_matrix(file_name_exchange):
    '''Reads in Quiver exchange matrix data and its class, and processes them into python lists, saving the data as floats''' 
    data = []
    em = open(file_name_exchange,'r') 
    data_em = em.readlines() 
    for i in data_em: 
        counter = 0
        n_removed = i.strip('\n')
        string_list = n_removed.split(',')
        for k in string_list: 
            test_value = float(k)
            string_list[counter] = test_value
            counter +=1 
        data.append(string_list)   
    em.close()    
    return data     


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
    elif setting == 'NONACYCLIC3': 
        bij = [[0,a,-c,f],[-a,0,b,e],[c,-b,0,d],[-f,-e,-d,0]]
    
    else: 
        bij = [[0,a,0,0],[-a,0,b,0],[0,-b,0,c],[0,0,-c,0]]
    
    return matrix(bij)    


def array_converter(matrix1): 
    '''INPUT: SAGEMATH QUIVER B MATRIX
       OUTPUT: PYTHON LIST
    '''
    array = matrix1.numpy()
    arraynewp1 = array.flatten() 
    arraynew = arraynewp1.astype('float64')   
    return arraynew.tolist()  


def listtoquiver(l):
    '''
    INPUT: list of flat 1-D lists
    OUTPUT: A list of SAGE Maths Quiver
    '''
    lister = Sequence_Iteration(l)
    output = []
    for k in lister:
        k_array = np.reshape(np.array(k,dtype='int'),(4,4)) 
        sage_matrix = matrix(k_array)
        q = ClusterSeed(sage_matrix).quiver()
        output.append(q)
    return output


# Import data
NMA_DATA = data_reading_exchange_matrix('Fourth_Experiment_NMA_data.txt')
NON_NMA_DATA = data_reading_exchange_matrix('Fourth_Experiment_MA_data.txt')
full_data = NMA_DATA + NON_NMA_DATA

# Output data sizes
print(len(full_data))
print(len(NMA_DATA))
print(len(NON_NMA_DATA))
print(len(NON_NMA_DATA+NMA_DATA))


ex1= full_data[1]

def components_finder_6(lists):
    '''
    INPUT: list of lists of 1 X 16 entries
    OUTPUT: lists of lists of 1 X 6 entries
    '''
    output =[]
    m = Sequence_Iteration(lists)
    for i in m: 
        outs = [i[1],i[2],i[3],i[6],i[7],i[11] ]
        output.append(outs)
    return output    


NON_NMA_DATA_NEW =components_finder_6(NON_NMA_DATA)
NMA_DATA_NEW = components_finder_6(NMA_DATA)
 
#################################################################################
# Support Vector Machine
full_data_removed = NON_NMA_DATA_NEW   #removed NMA Quivers.
seed_random = 30

# Class Builders 
not_nma_class = np.full(len(NON_NMA_DATA_NEW),-1)
nma_class = np.full(len(NMA_DATA_NEW),1)
cutoff_value = len(nma_class)

s_data = np.concatenate((NON_NMA_DATA_NEW,NMA_DATA_NEW))
s_class = np.concatenate((not_nma_class,nma_class))


s_data_2 = np.copy(s_data)
s_class_2 = np.copy(s_class)

np.random.seed(seed_random)
np.random.shuffle(s_data)
np.random.seed(seed_random)
np.random.shuffle(s_class)
np.random.seed(seed_random)
np.random.shuffle(s_data_2) 
np.random.seed(seed_random)
np.random.shuffle(s_class_2)

def equation_writer(equation,svm_polynomial_number): 
    '''
    INPUT: SAGE MATH Equation 
    OUTPUT: Writes to a txt. file the equation of that SVM_POLYNOMINAL_NUMBER 
    '''
    equation_string = str(latex(equation))
    print(equation_string) 
    file_name = 'Polynomial_Number_{}_SVM_EQUATION_Latex.txt'.format(int(svm_polynomial_number)) 
    files = open(file_name,'w')  
    files.write(equation_string)
    files.close()


def expression_length(expression):
    '''
    A function which takes SAGEMATH expression and returns how many terms there are in the expression
    INPUT: SAGEMATH EXPRESSION
    OUTPUT: Integer
    ''' 
    
    expanded_expression = expression.expand()
    expression_plus_sign_split = str(expanded_expression).split('+') 
    
    fully_split = []
    
    for i in Sequence_Iteration(expression_plus_sign_split):  
        minus_split = i.split('-')
        for j in minus_split: 
            fully_split.append(j)
        
    return len(fully_split)  


def Polynomial_Length(expression,MC,SVM_DEGREE):
    '''
    INPUT: (1) SAGEMATH EXPRESSION
           (2) Matthew's Coefficent
           (3) INTEGER: SVM_DEGREE
    OUTPUT: Writes the length of the polynomial and Matthew's Coeffcient to a file and returns it as a integer to the program
    '''
    
    length = expression_length(expression) 
    file_name = 'Number_of_Terms_in_SVM_Polynomial_Degree_{}.txt'.format(SVM_DEGREE)
    files = open(file_name,'w')
    files.write('Number of Terms: {}'.format(length))
    files.write('\n \n')
    files.write('Matthew Coefficient: {}'.format(MC))
    files.close()
    return length


proportion = float(len(NON_NMA_DATA_NEW)/len(NMA_DATA_NEW))

# Define and fit the SVM
svm_degree = 6
clf_6 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_6.fit(s_data,s_class)


#Matthew 
matt_poly_6 = matthews_corrcoef(s_class,clf_6.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_6))


#Quiver Classifier Function
def Quiver_Decider(quiver,svm_classifer): 
    '''
    A function which classifies where a quiver is Mutation acyclic (MA) or Non-Mutation Acyclic (NMA) depending on 
    a pretrained support vector machine (SVM). 
    INPUT: (1) SAGE MATH Quiver
           (2) Scikit Learn *trained* svm_classifer
    OUTPUT: Function returns type "None", but will print out to the console the string:
     "Quiver is {}", where {} can be:
     -Mutation Acyclic
     -Non Mutation Acyclic
    '''
    #Obtains Quiver's exchange matrix and converts it to a python list
    b_matrix_list = array_converter(quiver.b_matrix())
    
    #Gets the upper triangle of the exchange matrix for the quiver.
    quiver_upper_triangle = components_finder_6([b_matrix_list]) 

    #Gets the type of the Quiver
    quiver_type = svm_classifer.predict(np.array(quiver_upper_triangle))

    #Test if Mutation Acyclic (-1): 
    if quiver_type[0] == -1: 
        print('Quiver is Mutation Acyclic')
        return 
    else: 
        print('Quiver is Non Mutation Acyclic')
        return 
    
#EXAMPLE USE CASE

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

quiver_data = exchangematrix('NONACYCLIC2',a=2,b=2,c=2,d=2)
quiver = ClusterSeed(quiver_data).quiver()
Quiver_Decider(quiver,clf_6)    