'''
Script to perform PCA on the quiver adjacency matrix represenations, then appropriately compress. Followed by learning the mutation-acyclicity property via a support vector machine. 
'''
from sage.all import * 
import numpy as np
import sklearn.decomposition 
from sklearn import svm
from matplotlib import pyplot as plt
from itertools import combinations 
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
        #bij = [[0,a,0,d],[-a,0,d,0],[0,-d,0,-a],[-d,0,a,0]]
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


# Import the data
NMA_DATA = data_reading_exchange_matrix('Fourth_Experiment_NMA_data.txt')
NON_NMA_DATA = data_reading_exchange_matrix('Fourth_Experiment_MA_data.txt')
full_data = NMA_DATA+NON_NMA_DATA
# Output dataset sizes
print(len(full_data))
print(len(NMA_DATA))
print(len(NON_NMA_DATA))
print(len(NON_NMA_DATA+NMA_DATA))

#################################################################################
# Define and fit the PCA
FULL_DATA_ANALYSIS = sklearn.decomposition.PCA()  #Define class instance of PCA Class
FULL_DATA_ANALYSIS.fit(full_data)
eigenvalues = FULL_DATA_ANALYSIS.explained_variance_ratio_
print(eigenvalues)

# Identify the eigenvectors
for i in range(6):
    print('\t\t\t Eigenvector Number {}: \n {} \n'.format(i+1,FULL_DATA_ANALYSIS.components_[i]))

# Transform the datasets into the PCA basis
all_data_pca = FULL_DATA_ANALYSIS.transform(full_data)
nma_data_pca = FULL_DATA_ANALYSIS.transform(NMA_DATA)

# Define a meshgrid
list_of_combinations = combinations([0,1,2,3,4,5],2)
x = np.arange(0,3,1)
y =np.arange(0,5,1)
xv, yv = np.meshgrid(x, y, indexing='xy')
print(xv.flatten())
print(yv.flatten())


# Setup the PCA projection plotting
direction_list = ['First Direction','Second Direction','Third Direction','Fourth Direction','Fifth Direction','Sixth Direction']

figure, axis = plt.subplots(5, 3, figsize = [20,20])  
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.25, hspace=0.2)
figure.suptitle('PCA ANALYSIS FOR no removed isomorphic SECOND EXPERIMENT DATA', fontsize=30, verticalalignment ='top', horizontalalignment='center')  
z = zip(list_of_combinations,xv.flatten(),yv.flatten())
for i,x_value,y_value in z: 
    dir1, dir2 = i 
    axis[y_value,x_value].scatter(all_data_pca[:,dir1],all_data_pca[:,dir2],label='all')
    axis[y_value,x_value].scatter(nma_data_pca[:,dir1],nma_data_pca[:,dir2],label='nma')
    axis[y_value,x_value].set_xlabel(direction_list[dir1])
    axis[y_value,x_value].set_ylabel(direction_list[dir2])
    axis[y_value,x_value].legend(loc='best')
  
plt.savefig('PCA_SECOND_EXPERIMENT_NON_ISOMORPHIC.png')


#################################################################################
# Generic analysis of the PCA behaviour
# Investigating the Source of the Lines in the First Direction
#number of decimal points 
point_number = 4

# First Direction
first_data = np.round(all_data_pca[:,0],decimals=point_number)
first_direction_unique_entries = first_data
first_direction_unique_counts = np.unique(first_data,return_counts=True)
counts, bins = np.histogram(first_direction_unique_entries)
print(first_direction_unique_counts) 

plt.figure()
plt.title('First Direction Histogram')
plt.hist(first_direction_unique_entries, bins = first_direction_unique_counts[0]) # weights=first_direction_unique_counts[1])

 
# Second Direction
second_data = np.round(all_data_pca[:,1],decimals=point_number)
second_direction_unique_entries = np.round(second_data,decimals=3)
second_direction_unique_counts = np.unique(second_data,return_counts=True)
counts, bins = np.histogram(second_direction_unique_entries)
#print(second_direction_unique_counts)

plt.figure()
plt.title('Second Direction Histogram')
plt.hist(second_direction_unique_entries, bins = second_direction_unique_counts[0]) # weights=first_direction_unique_counts[1])

 
# Third Direction
Third_data = np.round(all_data_pca[:,2],decimals=point_number)
Third_direction_unique_entries = np.round(Third_data,decimals=3)
Third_direction_unique_counts = np.unique(Third_data,return_counts=True)
counts, bins = np.histogram(Third_direction_unique_entries)


plt.figure()
plt.title('Third Direction Histogram')
plt.hist(Third_direction_unique_entries, bins = Third_direction_unique_counts[0]) # weights=first_direction_unique_counts[1])

 
# Fourth Direction
Fourth_data = np.round(all_data_pca[:,3],decimals=point_number)
Fourth_direction_unique_entries = np.round(Fourth_data,decimals=3)
Fourth_direction_unique_counts = np.unique(Fourth_data,return_counts=True)
counts, bins = np.histogram(Fourth_direction_unique_entries)


plt.figure()
plt.title('Fourth Direction Histogram')
plt.hist(Fourth_direction_unique_entries, bins = Fourth_direction_unique_counts[0]) # weights=first_direction_unique_counts[1])

 
# Fifth Direction
Fifth_data = np.round(all_data_pca[:,4],decimals=point_number)
Fifth_direction_unique_entries = np.round(Fifth_data,decimals=3)
Fifth_direction_unique_counts = np.unique(Fifth_data,return_counts=True)
counts, bins = np.histogram(Fifth_direction_unique_entries)


plt.figure()
plt.title('Fifth Direction Histogram')
plt.hist(Fifth_direction_unique_entries, bins = Fifth_direction_unique_counts[0]) # weights=first_direction_unique_counts[1])


# Sixth Direction
Sixth_data = np.round(all_data_pca[:,5],decimals=point_number)
Sixth_direction_unique_entries = np.round(Sixth_data,decimals=3)
Sixth_direction_unique_counts = np.unique(Sixth_data,return_counts=True)
counts, bins = np.histogram(Sixth_direction_unique_entries)


plt.figure()
plt.title('Sixth Direction Histogram')
plt.hist(Sixth_direction_unique_entries, bins = Sixth_direction_unique_counts[0]) # weights=first_direction_unique_counts[1])


#################################################################################
# Checking Isomorphic Data for first Direction (to many dp)
first_direction_unique_counts =  np.unique(first_data,return_index=True)
print(first_direction_unique_counts[0])
print(first_direction_unique_counts[1])
 


#Finding the position for each value:

position_list = []
for i in range(len(first_direction_unique_counts[0])): 
    value = first_direction_unique_counts[0][i] 
    pos = np.where(first_data == value) 
    position_list.append(pos[0])

lowest_value = first_direction_unique_counts[0][0]
highest_value = first_direction_unique_counts[0][len(first_direction_unique_counts[0])-1]
print(lowest_value)
print(highest_value)

lowest_value_position = position_list[0][0]
highest_value_position = position_list[len(first_direction_unique_counts[0])-1][0]
print(lowest_value_position)
print(highest_value_position)

high_list = full_data[lowest_value_position] 
low_list = full_data[highest_value_position]


q6 = listtoquiver([low_list,high_list])
for u in q6:
    u.show()


print(high_list)
print(low_list)


################################################################################# 
# Support Vector Machine
full_data_array = np.array(full_data.copy(),dtype =int) 
print(full_data_array.shape)
full_data_removed = NON_NMA_DATA   #removed NMA Quivers.

# Creating a transformation matrix 
projection_matrix = np.vstack((FULL_DATA_ANALYSIS.components_[0],FULL_DATA_ANALYSIS.components_[1],FULL_DATA_ANALYSIS.components_[2],FULL_DATA_ANALYSIS.components_[3],FULL_DATA_ANALYSIS.components_[4],FULL_DATA_ANALYSIS.components_[5]))

new_NMA_removed_full_data = [] 
new_NMA = []

for i in range(len(full_data_removed)):
    new_vector = np.matmul(projection_matrix, full_data_removed[i]) 
    new_NMA_removed_full_data.append(new_vector)   
        
print(new_NMA_removed_full_data[0])
print(new_NMA_removed_full_data[1])

for i in range(len(NMA_DATA)):
    new_vector = np.matmul(projection_matrix, NMA_DATA[i]) 
    new_NMA.append(new_vector)  

print(len(new_NMA))
print(new_NMA[0])
print(new_NMA[1])


# Class Builders 
not_nma_class = np.full(len(NON_NMA_DATA),-1)
nma_class = np.full(len(NMA_DATA),1)
cutoff_value = len(nma_class)

s_data = np.concatenate((new_NMA_removed_full_data,new_NMA))
s_class = np.concatenate((not_nma_class,nma_class))
print(s_data.shape)

s_data_2 = np.copy(s_data)
s_class_2 = np.copy(s_class)


np.random.seed(10)
np.random.shuffle(s_data)
np.random.seed(10)
np.random.shuffle(s_class)
np.random.seed(10)
np.random.shuffle(s_data_2) 
np.random.seed(10)
np.random.shuffle(s_class_2)

s_data == s_data_2
s_class == s_class_2




proportion = float(len(NON_NMA_DATA)/len(NMA_DATA))

# Define and fit the SVM
clf = svm.SVC(kernel ='poly', C=1,degree = 6,class_weight = {-1:1,1:proportion})
clf.fit(s_data,s_class)

clf.get_params(deep=True)
df = clf.decision_function(s_data)
rang = np.arange(0,len(df))

print(len(rang[14682:15105]))#
print(len(df[14682:15105]))

# Plot the learnt SVM function
plt.figure()
plt.scatter(rang,df)
plt.scatter(rang[14682:15104],df[14682:15104])

np.random.seed(10)
np.random.shuffle(rang)
x_rnt = rang[14682:15104]
plt.figure()
plt.scatter(rang,df)
plt.scatter(x_rnt,df[14682:15104])

# Test the SVM
tests = clf.predict(s_data) == s_class 
print(len(tests))
print(tests)
r = np.where(tests == True)[0] 
print(len(r))

#Matthew 
matt_poly_6 = matthews_corrcoef(s_class,clf.predict(s_data))
print('\t\t\t\t\t\tThe Matthews Coefficient is: {}'.format(matthews_corrcoef(s_class,clf.predict(s_data))))

print('Percentage Correct: {}'.format((len(r)/len(tests))*100))

tests_1_NMA = clf.predict(new_NMA) == np.full(len(new_NMA),1) 
print(len(tests_1_NMA))
print(tests_1_NMA)
r = np.where(tests_1_NMA == True)[0] 
print(len(r))

print('Percentage NMA Correct: {}'.format((len(r)/len(tests_1_NMA))*100))


#Linear Test
clf2 = svm.SVC(kernel ='linear',class_weight = {-1:1,1:proportion},C=1)
clf2.fit(s_data,s_class)
 
tests2 = clf2.predict(s_data) == s_class 
print(len(tests2))
print(tests2)
r2 = np.where(tests2 == True)[0] 
print(len(r2))
print('Percentage Correct: {}'.format((len(r2)/len(tests2))*100))

#Matthew 
matt_poly_1 = matthews_corrcoef(s_class,clf2.predict(s_data))
print('The Matthews Coefficient is: {}'.format(matthews_corrcoef(s_class,clf2.predict(s_data))))

sv2 = clf2.support_vectors_
print(len(sv2))


#Linear_NMA_Correct 

linear_nma_data = tests2[len(tests2)-cutoff_value:len(tests2)] 
rl = np.where(linear_nma_data == True)[0] 
print(len(rl))
print('Percentage NMA Correct: {}'.format((len(rl)/len(linear_nma_data))*100))

df_linear = clf2.decision_function(s_data)
rang_linear = np.arange(0,len(df_linear))

plt.figure()
plt.title('Linear Decision Function')
plt.scatter(rang_linear,df_linear)
plt.scatter(rang_linear[14682:15104],df_linear[14682:15104])

plt.figure()
plt.title('Linear Decision Function')
plt.scatter(rang,df_linear)
plt.scatter(x_rnt,df_linear[14682:15104])


#Quadratic Test Test
clf3 = svm.SVC(kernel ='poly',degree = 2,class_weight = {-1:1,1:proportion},C=1) #ONLY WORKS USING POLY and C=1000 class_weight='balanced'
clf3.fit(s_data,s_class)

# tests = clf.predict(s_data) == s_class 
tests3= clf3.predict(s_data) == s_class 
print(len(tests3))
print(tests3)
r3 = np.where(tests3 == True)[0] 
print(len(r3))
print('Percentage Correct: {}'.format((len(r3)/len(tests3))*100))

#Matthew 
matt_poly_2 = matthews_corrcoef(s_class,clf3.predict(s_data))
print('Matthews Coefficient is: {}'.format(matthews_corrcoef(s_class,clf3.predict(s_data))))


#Cubic Test Test
clf5 = svm.SVC(kernel ='poly',degree = 3,class_weight = {-1:1,1:proportion},C=1) #ONLY WORKS USING POLY and C=1000 class_weight='balanced'
clf5.fit(s_data,s_class)

tests5= clf5.predict(s_data) == s_class 
print(len(tests5))

r5 = np.where(tests5 == True)[0] 
print(len(r5))
print('Percentage Correct: {}'.format((len(r5)/len(tests5))*100))

matt_poly_3 = matthews_corrcoef(s_class,clf5.predict(s_data))
print('Matthews Coefficient is: {}'.format(matthews_corrcoef(s_class,clf5.predict(s_data))))


#Quartic Test Test
clf6 = svm.SVC(kernel ='poly',degree = 4,class_weight = {-1:1,1:proportion},C=1) 
clf6.fit(s_data,s_class)

tests6= clf6.predict(s_data) == s_class 
print(len(tests6))

r6 = np.where(tests6 == True)[0] 
print(len(r6))
print('Percentage Correct: {}'.format((len(r6)/len(tests6))*100))

matt_poly_4 = matthews_corrcoef(s_class,clf6.predict(s_data))
print('Matthews Coefficent is: {}'.format(matthews_corrcoef(s_class,clf6.predict(s_data))))


#Quintic Test 
clf7 = svm.SVC(kernel ='poly',degree = 5,class_weight = {-1:1,1:proportion},C=1) 
clf7.fit(s_data,s_class)
 
tests7= clf7.predict(s_data) == s_class 
print(len(tests7))

r7 = np.where(tests7 == True)[0] 
print(len(r7))
print('Percentage Correct: {}'.format((len(r7)/len(tests7))*100))

matt_poly_5 = matthews_corrcoef(s_class,clf7.predict(s_data))
print('Matthews Coefficient is: {}'.format(matthews_corrcoef(s_class,clf7.predict(s_data))))


#RBF
clf4 = svm.SVC(kernel ='rbf',class_weight = {-1:1,1:proportion},C=20) 
clf4.fit(s_data,s_class)
 
tests4= clf4.predict(s_data) == s_class 
print(len(tests4))
print(tests4)
r4 = np.where(tests4 == True)[0] 
print(len(r4))
print('Percentage Correct: {}'.format((len(r4)/len(tests4))*100))
print(clf4.predict(s_data))

tests_1_NMA = clf4.predict(new_NMA) == np.full(len(new_NMA),1) 
print(len(tests_1_NMA))

r = np.where(tests_1_NMA == True)[0] 
print(len(r))

print('Percentage NMA Correct: {}'.format((len(r)/len(tests_1_NMA))*100))
print('Matthews Coefficent is: {}'.format(matthews_corrcoef(s_class,clf4.predict(s_data)))) 

#plot matthew coeffs
plt.figure()
plt.title('Matthew Coefficient vs Polynomial Kernal Order ')
plt.xlabel('Polynomial Order')
plt.ylabel('Matthew Coefficient')
plt.grid()
plt.scatter([1,2,3,4,5,6],[matt_poly_1,matt_poly_2,matt_poly_3,matt_poly_4,matt_poly_5,matt_poly_6])
plt.show()
