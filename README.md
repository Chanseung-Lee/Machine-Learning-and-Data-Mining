# Introduction

This project summary encompasses the several implementations of mine surrounding concise yet intuitive machine learning concepts. The data used for each implementation have been simplified for both computational runtime and general comprehension purposes, however their key variances have been preserved even in cases where some of the dimensions have been reduced drastically. 

# Table of Contents

- [kNN Classification](#k-nearest-neighbor-classification-and-statistical-estimation)
- [Linear Model Regression & Classification](#linear-models-for-regression-and-classification)
- [Neural Network](#simple-neural-network)
- [Clustering](#k-means-clustering)

# k-Nearest Neighbor Classification and Statistical Estimation

We are considering a binary classification problem where the goal is to classify whether a person has an annual income more or less than $50,000 given census information. 

Below is a table listing the attributes available in the dataset. It's worth noting that categorizing some of these attributes into two or a few categories is reductive (e.g. only 14 occupations) or might reinforce a particular set of social norms (e.g. categorizing sex or race in particular ways).

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/2106e7a73040aefa82863cd60a0e1bb8e9152d80/f.JPG" alt="Image" width="750" height="480"></kbd>

Our dataset has three types of attributes – numerical, ordinal, and nominal. For nominal variables like workclass, marital-status, occupation, relationship, race, and native-country, we’ve transformed these into one column for each possible value with either a 0 or a 1. For example, the first instance in the training set reads: [0, 0.136, 0.533, 0.0, 0.659, 0.397, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] where all the zeros and ones correspond to these binarized variables.

Now to make the KNN predictions, which is straightforward because we already have preprocessed data. To make a prediction for a new data point, there are three main
steps:

1. Calculate the distance of that data point to every training example.
2. Get the k nearest points (i.e. the k with the lowest distance).
3. Return the majority class of these neighbors as the prediction

Implementing this using native Python operations will be quite slow, but lucky for us there is the numpy package.
numpy makes matrix operations quite efficient and easy to code.

In kNN, we need to compute distance between every training example xi and a new point z in order to make a prediction. Computing this for one xi can be done by applying an arithmetic operation between xi and z, then taking a norm. In numpy, arithmetic operations between matrices and vectors are sometimes defined by “broadcasting”, even if standard linear algebra doesn’t allow for them. For example, given a matrix X of size n × d and a vector z of size d, numpy will happily compute Y = X − z such that Y is the result of the vector z being subtracted from each row of X.

Class labels for the test set or a validation set are not provided, so it will be optimal to implement K-fold Cross Validation to check if our implementation is correct and to select hyperpameters. K-fold Cross Validation divides the training set into K segments and then trains K models – leaving out one of the segments each time and training on the others. Then each trained model is evaluated on the left-out fold to estimate performance. Overall performance is estimated as the mean and variance of these K fold performances.

Finally, code at the end of main() outputs predictions for the test set to test_predicted.csv. Hyperparameters can be adjusted for higher prediction accuracy.

# Linear Models for Regression and Classification

The goal is to implement a logistic regression model for predicting whether a tumor is malignant (cancerous) or benign (non-cancerous). The dataset has eight attributes – clump thickness, uniformity of cell size, uniformity of cell shape, marginal adhesion, single epithelial cell size, bland chromatin, nomral nucleoli, and mitoses – all rated between 1 and 10.

The logistic regression algorithm is a binary classifier that learns a linear decision boundary. Specifically, it predicts the probability of an example x ∈ R d to be class 1 as:

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/6fa5674951c4faa681b789d76b5e5e2cc3efd0e0/r.JPG" alt="Image"></kbd>



# Simple Neural Network



# k-Means Clustering

