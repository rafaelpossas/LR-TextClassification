# Assignment 1 Machine Learning and Data Mining
Project for assignment 1 of Machine Learning and Data Mining course

This project has 4 main folders:

1. algorithm: python classes and functions for solving the problem

2. input: empty folder to place files

3. output: folder where the final prediction is

4. report: folder with latex files to create the report

## How to run the code

In order to run the code you go to the algorithm folder and run

```bash
python Main.py
```
This will run the default mode of the project. This mode use 0.3 as regularization
parameter, predict the labels for the test data and save the results in output folder
 and will not use any dimensionality reduction process.

If you want to use other modes you can run the project with parameters:

 ```bash
python Main.py l p r --value_reduction x --histo
 ```
- l is the regularization value
- p is the process to do (cross: run the 10-fold
cross-validation with the specific parameters; test: run a testing over a 1/3 of the data,
training with the rest 2/3 of the data and will show a confusion matrix;
predict: run a prediction for the test data, using all the data for training).
- r is the dimensionality reduction process (common: reduce the data matrix just chosing
the columns with less than x rows with values, pca: apply pca to the data matrix using
x as a parameter of n_components, none: do not apply any pre-processing)
- x is the value to use in dimensionality reduction
- if you place --histo in the run command, just a histogram of the distribution of the
labels in training data is showed, and the execution does not continue.
