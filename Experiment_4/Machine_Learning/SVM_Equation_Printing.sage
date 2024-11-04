'''Code to print the polynomial equation of a trained SVM classifier'''
# Import libraries
import numpy as np
import pickle
import sklearn

# Define import functions
class Sequence_Iteration:
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

def components_finder_6(lists):
    output =[]
    m = Sequence_Iteration(lists)
    for i in m: 
        outs = [i[1],i[2],i[3],i[6],i[7],i[11]]
        output.append(outs)
    return output    

# Import data used
filepath = '~/github/Experiment_4/Machine_Learning/' #...edit accordingly
NMA_DATA = data_reading_exchange_matrix(filepath+'Fourth_Experiment_NMA_data.txt')
NON_NMA_DATA = data_reading_exchange_matrix(filepath+'Fourth_Experiment_MA_data.txt')
NON_NMA_DATA_NEW =components_finder_6(NON_NMA_DATA)
NMA_DATA_NEW = components_finder_6(NMA_DATA)
data = np.concatenate((NON_NMA_DATA_NEW,NMA_DATA_NEW))

# Load pre-trained model (ensure the filepath to the model is correct)
with open(filepath+'trained_svm_model.pkl', 'rb') as f:
    svm_clf = pickle.load(f)

# Print the number of support vectors:
print(f'Number of Support Vectors: {svm_clf.n_support_}')

# Equation properties in the embedding space
input_dim = 6 #...upper triangle of rank 4 adjacency matrices
svm_degree = svm_clf.degree
Coefficients = svm_clf.dual_coef_[0]
SupportVectors = svm_clf.support_vectors_
intercept = svm_clf.intercept_[0]
gamma = 1./(data.var()*data.shape[1])
#svm_clf.coef0 == 0., svm_clf.gamma == 'scale'

# Generic decision function (hyperplane where this = 0)
def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept

# Define the generic input variables
symbolic_input = var('x', n=input_dim, latex_name='x')

# Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept
print(equation.full_simplify())             
