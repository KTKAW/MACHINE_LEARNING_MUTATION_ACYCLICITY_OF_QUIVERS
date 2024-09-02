'''
Script to predict mutation-acyclicity of an input quiver (using their adjacency upper triangular representation).
The prediction is performed by a degree-6 via support vector machines.
To test a quiver of your choice modify the adjacency matrix in line 17, and run the script.
'''

# Import libraries
import numpy as np 
import pickle
import sklearn

# Load pre-trained model (ensure the filepath to the model is correct)
with open('trained_svm_model.pkl', 'rb') as f:
    pretrained_clf = pickle.load(f)

# Define the test quiver (from adjacency matrix)
test_quiver = [[0,1,0,0],[-1,0,1,0],[0,-1,0,1],[0,0,-1,0]] #...EDIT HERE

# Classify the test quiver
test_quiver_rep = np.array(test_quiver[0][1:]+test_quiver[1][2:]+test_quiver[2][3:]) #...take the upper triangle of the adjacency
prediction = pretrained_clf.predict([test_quiver_rep])

#Converts prediction to a boolean type
if prediction[0] == -1: 
    prediction_bool = True  
else: 
    prediction_bool = False

# Output the prediction
print(f'The quiver with adjacency matrix:\n{test_quiver}\nis mutation-acyclic? --> {prediction_bool}')






