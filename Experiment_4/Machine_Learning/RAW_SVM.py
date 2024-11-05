'''
Script to learn mutation-acyclicity of quivers (using their adjacency upper triangular representation), via support vector machines.
'''
# Import libraries
import numpy as np
from sage.all import * 
from sklearn import svm
from matplotlib import pyplot as plt 
from sklearn.metrics import matthews_corrcoef
# Save using pickle 
import pickle


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
print(s_data.shape)

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

clf_6.get_params(deep=True)
df = clf_6.decision_function(s_data)
rang = np.arange(0,len(df))

plt.figure()
plt.scatter(rang,df)
plt.scatter(rang[14682:15104],df[14682:15104])

np.random.seed(seed_random)
np.random.shuffle(rang)
x_rnt = rang[14682:15104]
plt.figure()
plt.scatter(rang,df)
plt.scatter(x_rnt,df[14682:15104])

# Test the SVM
tests = clf_6.predict(s_data) == s_class 
print(len(tests))
print(tests)
r = np.where(tests == True)[0] 
print(len(r))

#Matthew 
matt_poly_6 = matthews_corrcoef(s_class,clf_6.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_6))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_6.n_support_}')
input_dim = 6
#Equation properties in the embedding space
Coefficients = clf_6.dual_coef_[0]
SupportVectors = clf_6.support_vectors_
intercept = clf_6.intercept_[0]
gamma = 1./(s_data.var()*input_dim)
#Generic decision function (hyperplane where this = 0)
def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept

#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e') 

print('Gamma IS:',gamma)
print('INTERCEPT IS:',intercept)

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = 1e3*( sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)
print(equation.expand())      

equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_6,svm_degree)

# save the model using "pickle" package
# with open('trained_svm_model.pkl','wb') as f:
#     pickle.dump(clf_6,f)

svm_degree = 5
clf_5 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_5.fit(s_data,s_class)

input_dim = 6

#Matthew 
matt_poly_5 = matthews_corrcoef(s_class,clf_5.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_5))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_5.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_5.dual_coef_[0]
SupportVectors = clf_5.support_vectors_
intercept = clf_5.intercept_[0]
gamma = 1./(s_data.var()*input_dim)
#Generic decision function (hyperplane where this = 0)
def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e15
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_5,svm_degree)

svm_degree = 4
clf_4 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})

clf_4.fit(s_data,s_class)



#Matthew 
matt_poly_4 = matthews_corrcoef(s_class,clf_4.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_4))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_4.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_4.dual_coef_[0]
SupportVectors = clf_4.support_vectors_
intercept = clf_4.intercept_[0]
gamma = 1./(s_data.var()*input_dim)
#Generic decision function (hyperplane where this = 0)
def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e2
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_4,svm_degree)

svm_degree = 3
clf_3 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_3.fit(s_data,s_class)

#Matthew 
matt_poly_3 = matthews_corrcoef(s_class,clf_3.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_3))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_3.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_3.dual_coef_[0]
SupportVectors = clf_3.support_vectors_
intercept = clf_3.intercept_[0]
gamma = 1./(s_data.var()*input_dim)
#Generic decision function (hyperplane where this = 0)
def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e14
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_3,svm_degree)

svm_degree = 2
clf_2 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_2.fit(s_data,s_class)


#Matthew 
matt_poly_2 = matthews_corrcoef(s_class,clf_2.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_2))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_2.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_2.dual_coef_[0]
SupportVectors = clf_2.support_vectors_
intercept = clf_2.intercept_[0]
gamma = 1./(s_data.var()*input_dim)
def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e2
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_2,svm_degree)

svm_degree = 1
clf_1 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_1.fit(s_data,s_class)

#Matthew 
matt_poly_1 = matthews_corrcoef(s_class,clf_1.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_1))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_1.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_1.dual_coef_[0]
SupportVectors = clf_1.support_vectors_
intercept = clf_1.intercept_[0]
gamma = 1./(s_data.var()*input_dim)

def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e14 
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_1,svm_degree)

##################################################################################
#Higher Order Even Polynomials# 

#8 

svm_degree = 8
clf_8 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_8.fit(s_data,s_class)

#Matthew 
matt_poly_8 = matthews_corrcoef(s_class,clf_8.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_8))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_8.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_8.dual_coef_[0]
SupportVectors = clf_8.support_vectors_
intercept = clf_8.intercept_[0]
gamma = 1./(s_data.var()*input_dim)

def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e14 
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_8,svm_degree)

# 10

svm_degree = 10
clf_10 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_10.fit(s_data,s_class)

#Matthew 
matt_poly_10 = matthews_corrcoef(s_class,clf_10.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_10))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_8.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_10.dual_coef_[0]
SupportVectors = clf_10.support_vectors_
intercept = clf_10.intercept_[0]
gamma = 1./(s_data.var()*input_dim)

def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e14 
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_10,svm_degree)



# 12

svm_degree = 12
clf_12 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_12.fit(s_data,s_class)

#Matthew 
matt_poly_12 = matthews_corrcoef(s_class,clf_12.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_12))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_12.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_12.dual_coef_[0]
SupportVectors = clf_12.support_vectors_
intercept = clf_12.intercept_[0]
gamma = 1./(s_data.var()*input_dim)

def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e14 
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_12,svm_degree)


# 14

svm_degree = 14
clf_14 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_14.fit(s_data,s_class)

#Matthew 
matt_poly_14 = matthews_corrcoef(s_class,clf_14.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_14))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_14.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_14.dual_coef_[0]
SupportVectors = clf_14.support_vectors_
intercept = clf_14.intercept_[0]
gamma = 1./(s_data.var()*input_dim)

def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e14 
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_14,svm_degree)



# 16

svm_degree = 16
clf_16 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_16.fit(s_data,s_class)

#Matthew 
matt_poly_16 = matthews_corrcoef(s_class,clf_16.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_16))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_16.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_16.dual_coef_[0]
SupportVectors = clf_16.support_vectors_
intercept = clf_16.intercept_[0]
gamma = 1./(s_data.var()*input_dim)

def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e14 
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_16,svm_degree)

# 18

svm_degree = 18
clf_18 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_18.fit(s_data,s_class)

#Matthew 
matt_poly_18 = matthews_corrcoef(s_class,clf_18.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_18))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_18.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_18.dual_coef_[0]
SupportVectors = clf_18.support_vectors_
intercept = clf_18.intercept_[0]
gamma = 1./(s_data.var()*input_dim)

def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e14 
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_18,svm_degree)


with open('trained_svm_model.pkl','wb') as f:
    pickle.dump(clf_18,f)


# 20

svm_degree = 20
clf_20 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_20.fit(s_data,s_class)

#Matthew 
matt_poly_20 = matthews_corrcoef(s_class,clf_20.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_20))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_20.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_20.dual_coef_[0]
SupportVectors = clf_20.support_vectors_
intercept = clf_20.intercept_[0]
gamma = 1./(s_data.var()*input_dim)

def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e14 
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_20,svm_degree)


# 22

svm_degree = 22
clf_22 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_22.fit(s_data,s_class)

#Matthew 
matt_poly_22 = matthews_corrcoef(s_class,clf_22.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_22))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_22.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_22.dual_coef_[0]
SupportVectors = clf_22.support_vectors_
intercept = clf_22.intercept_[0]
gamma = 1./(s_data.var()*input_dim)

def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e14 
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_22,svm_degree)


# 24

svm_degree = 24
clf_24 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_24.fit(s_data,s_class)

#Matthew 
matt_poly_24 = matthews_corrcoef(s_class,clf_24.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_24))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_24.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_24.dual_coef_[0]
SupportVectors = clf_24.support_vectors_
intercept = clf_24.intercept_[0]
gamma = 1./(s_data.var()*input_dim)

def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e14 
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_24,svm_degree)



# 26

svm_degree = 26
clf_26 = svm.SVC(kernel ='poly', C=1,degree = svm_degree,class_weight = {-1:1,1:proportion})
clf_26.fit(s_data,s_class)

#Matthew 
matt_poly_26 = matthews_corrcoef(s_class,clf_26.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Constant is: {}'.format(matt_poly_26))

#Print the number of support vectors:
print(f'Number of Support Vectors: {clf_26.n_support_}')

#Equation properties in the embedding space
Coefficients = clf_26.dual_coef_[0]
SupportVectors = clf_26.support_vectors_
intercept = clf_26.intercept_[0]
gamma = 1./(s_data.var()*input_dim)

def DecisionFunction(InputVector):
    return sum([(gamma*np.dot(SupportVectors[i], InputVector))**svm_degree * Coefficients[i] for i in range(len(Coefficients))]) + intercept


#Define the generic input variables
input_dim = 6
symbolic_input = var('e', n=input_dim, latex_name='e')

#Define the symbolic equation (note where this = 0 defines the hyperplane)
equation = (sum([(gamma*sum(SupportVectors[j,i]*symbolic_input[i] for i in range(input_dim)))**svm_degree*Coefficients[j] for j in range(len(SupportVectors))]) + intercept)*1e14 
equation_writer(equation.expand(),svm_degree)
Polynomial_Length(equation,matt_poly_26,svm_degree)
