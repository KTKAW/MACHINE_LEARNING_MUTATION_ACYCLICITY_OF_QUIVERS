'''
Script to learn mutation-acyclicity of quivers (using their adjacency upper triangular representation), via support vector machines.
'''

# Import libraries
import numpy as np 
import pickle
import sklearn

# Load pre-trained model
with open('model.pkl', 'rb') as f:
    pretrained_clf = pickle.load(f)

# Define the test quiver (from adjacency matrix)
test_quiver = [[0,1,0,0],[-1,0,1,0],[0,-1,0,1],[0,0,-1,0]]


# Classify the test quiver
test_quiver_rep = np.array(test_quiver[0][1:]+test_quiver[1][2:]+test_quiver[2][3:]) #...take the upper triangle of the adjacency
prediction = pretrained_clf.predict([test_quiver_rep])

#Converts prediction to a boolean type
if prediction[0] == -1: 
    prediction_bool = 1 #true
else: 
    prediction_bool  = 0  #false


# Output the prediction
print(f'The quiver with adjacency matrix:\n{test_quiver}\nis mutation-acyclic? --> {bool(prediction_bool)}')






