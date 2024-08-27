'''
Script to perform NN classification between supersets of quiver mutation classes separated by the mutation-acyclicity property.
'''
# Import libraries
from sage.all import *
import numpy as np
import tensorflow as tf 
import sklearn 
import sklearn.preprocessing
import sklearn.model_selection 
from sklearn.metrics import matthews_corrcoef
from matplotlib import pyplot as plt 
import random 

# Define appropriate functions
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

def data_reading_catagories(file_name_cat): 
    '''Reads in classes for the exchange matrices''' 
    data = []
    cat = open(file_name_cat,'r')
    data_cat = cat.readlines()
    for i in data_cat:
        counter = 0
        n_removed = i.strip('\n')
        number = float(n_removed) 
        data.append(number)
        counter += 1 
    cat.close()    
    return data 


def class_balancing(MA,NMA):
    '''Takes in lists, finds their length. Finds the list with smallest length and returns the smallest number'''
    MA_length = len(MA)
    NMA_length = len(NMA)
    
    length_array = np.array([MA_length,NMA_length])
    print('Length_Array',length_array)
    smallest_number = int(np.min(length_array))
    return smallest_number


# Define the filepaths
MA_name = 'MA_ALL_DEPTH_Original_6_type_0.txt' 
MA_name_cat = 'MA_ALL_DEPTH_Original_6_type_0_cat.txt' 

MA_bm = data_reading_exchange_matrix(MA_name)
MA_bm_2 = random.sample(MA_bm,len(MA_bm))
MA_name_cat = data_reading_catagories(MA_name_cat) 

NMA_name = 'NMA_ALL_Depth_Original_6__type_1.txt' 
NMA_name_cat = 'NMA_ALL_Depth_Original_6__type_1_cat.txt'

NMA_bm = data_reading_exchange_matrix(NMA_name)
NMA_bm_2 = random.sample(NMA_bm,len(NMA_bm))
NMA_name_cat = data_reading_catagories(NMA_name_cat) 

# Balance the data classes
splicing = class_balancing(MA_bm_2,NMA_bm_2)
print('Splicing',splicing)

all_bm = MA_bm_2[0:splicing]+NMA_bm_2[0:splicing]
all_data_classes = MA_name_cat[0:splicing]+NMA_name_cat[0:splicing]

print('Amount of data points:',len(all_bm)) 

all_bm_array = np.array(all_bm)
all_data_classes_array = np.array(all_data_classes) 
all_bm_rescaled = all_bm_array * 0.00001 
all_data_classes_rescaled = all_data_classes_array

# Setup the data for ML
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(all_bm_rescaled,all_data_classes_rescaled,test_size = float(0.1),shuffle = True)

# Define the ML hyperparameters
Nepochs = 10000 
learning_rate  = 0.0001 
validation = 0.3
batch = 200
dropout = 0.4

# Define the NN model
model = tf.keras.models.Sequential() 

model.add(tf.keras.layers.Input(shape=(16,)))
model.add(tf.keras.layers.Dense(128,activation='selu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(128,activation='selu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(128,activation='tanh'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(128,activation='selu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(2,activation='softmax'))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy() 
opt = tf.optimizers.Adam(learning_rate = learning_rate,clipnorm=10.0) 

model.compile(loss=loss_fn,optimizer=opt,metrics=['accuracy'])

model.summary() 

# Train the model
history = model.fit(x_train,y_train,validation_split=validation,batch_size=batch,epochs=Nepochs,shuffle=True) 

loss = history.history['loss'] 
val_loss = history.history['val_loss']
acc_tr = history.history['accuracy']
acc_val =history.history['val_accuracy'] 

##############################################
#Print Stats to File
##############################################
loss_save = open('Loss.txt','w')
loss_save.write(str(loss) )
loss_save.close()

val_loss_save = open('Val_Loss.txt','w')
val_loss_save.write(str(val_loss)) 
val_loss_save.close()

acc_tr_save = open('Acc.txt','w')
acc_tr_save.write(str(acc_tr) )
acc_tr_save.close()

val_acc_save = open('Val_Acc.txt','w')
val_acc_save.write(str(acc_val) )
val_acc_save.close()

###################################################
#Graph Print
##################################################

plt.figure()
plt.title('MLP Loss')
plt.plot(loss,'r-',label='Training')
plt.plot(val_loss,'b-',label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig('MLP_Loss.jpg')

plt.figure()
plt.plot(acc_tr,'r-',label='Training')
plt.plot(acc_val,'b-',label='Validation')
plt.title('MLP Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('MLP_Accuracy.jpg')

# Test the model
prediction = model.predict(x_test) 
predicts_max = np.argmax(prediction,axis=1) 
y_test_int = y_test.astype('i') #converts out y_test data from floats to integers 
bools = predicts_max == y_test_int 
length_of_bool = len(bools) 
number_of_correct = np.count_nonzero(bools== True) 
percent_correct = (number_of_correct/length_of_bool) *100 

string = 'Percentage Correctly Predicted: {} %\n' 
files = open('Correct.txt','w')
files.write(string.format(percent_correct))
files.write('Matthew_Coefficient: {}'.format(matthews_corrcoef(y_test_int,predicts_max)))
files.close()

