# Machine Learning Mutation-Acyclicity of Quivers
This repository contains code scripts for data generation and machine learning of quiver mutation classes. The workflow is split into 4 experiments, the first 3 use neural networks to classify between mutation classes directly, and then between supersets of mutation classes expressing mutation-acyclicity or not. The final experiment uses support vector machines to learn this mutation-acyclicity property, outputting a directly interpretable equation format (here PCA methods are also used to confirm the data representation compression).

------------------------------------------------------------------------

Included in this repository, within `Experiment_4/Machine_Learning/`, is the saved degree-6 kernel Support Vector Machine model that exhibited perfect test performances, `trained_svm_model.pkl`.     
...the script `Pretrained_Quiver_Classifier.py` then provides functionality for an interested reader to enter a rank 4 quiver of their choice (as an adjacency matrix), and then output the trained SVM prediction of whether it is mutation-acyclic or not.

Additions: computer-assissted proof.

------------------------------------------------------------------------
# BibTeX Citation
``` 
Raise NotImplementedError
```
