# Machine Learning Mutation-Acyclicity of Quivers
This repository contains code scripts for data generation and machine learning of quiver mutation classes. The workflow is split into 4 experiments, the first 3 use neural networks to classify between mutation classes directly, and then between supersets of mutation classes expressing mutation-acyclicity or not. The final experiment uses support vector machines to learn this mutation-acyclicity property, outputting a directly interpretable equation format (here PCA methods are also used to confirm the data representation compression).

------------------------------------------------------------------------

Included in this repository, within `Experiment_4/Machine_Learning/`, is the saved degree-18 kernel Support Vector Machine model that exhibited optimum test performances, `trained_svm_model.pkl`.     
The script `Pretrained_Quiver_Classifier.py` then provides functionality for an interested reader to enter a rank 4 quiver of their choice (as an adjacency matrix), and then output the trained SVM prediction of whether it is mutation-acyclic or not.

The directory `Theorem_2_8` contains scripts for the computer-assissted proof of the paper's Theorem 2.8, exhaustively classifying all isomorphism classes of quivers on 4 vertices with edge multiplicities $<3$ according to mutation-acyclicity.   

We note many of the scripts rely on [sagemath](https://www.sagemath.org/) kernels, which can be setup as describe in their documentation. Quivers are then represented and mutated using the [Cluster Algebra](https://arxiv.org/abs/1102.4844) package.

------------------------------------------------------------------------
# BibTeX Citation
``` 
@article{Armstrong-Williams:2024nzy,
    author = "Armstrong-Williams, Kymani T. K. and Hirst, Edward and Jackson, Blake and Lee, Kyu-Hwan",
    title = "{Machine Learning Mutation-Acyclicity of Quivers}",
    eprint = "2411.04209",
    archivePrefix = "arXiv",
    primaryClass = "math.CO",
    reportNumber = "QMUL-PH-24-27",
    month = "11",
    year = "2024"
}
```
