# Introduction

This project summary encompasses the several implementations of mine that surround concise yet intuitive machine learning concepts. The data used for each implementation have been simplified for both computational runtime and general comprehension purposes, however their key variances have been preserved even in cases where some of the dimensions have been reduced drastically. 

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

where w ∈ R d is a weight vector that we want to learn from data. To estimate these parameters from a dataset of n input-output pairs D, we assumed
yi ∼ Bernoulli(θ = σ(wT xi)) and wrote the negative log-likelihood:

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/a975dd26c6f0427178441e133c3d094fe646ba04/yu.JPG" alt="Image"></kbd>

We want t find optimal weights w∗ = argminw − logP(D|w). However, taking the gradient of the negative log-likelihood yields the expression below which does not offer a closed-form solution.

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/0d49384af71d88cebcb5c46bbe394a07d72e4975/54.JPG" alt="Image"></kbd>

Instead, we opted to minimize −logP(D|w) by gradient descent.

1. Initialize w to some initial vector (all zeros,random, etc)
2. Repeat until max iterations:
  (a) w = w − α ∗ ∇w(−logP(D|w))
  
For convex functions (and sufficiently small values of the stepsize α), this will converge to the minima.
We can also express this as a product between a matrix (X) and a vector of these errors. Specifically, assuming the logistic function σ(·) is applied elementwise when given a vector, we could compute:

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/8e40fab98b0acd3f5142754ca0dfc9c5b20076ec/dwdw.JPG" alt="Image"></kbd>

Now to add a bias term, which allows the model to "unpin" its decision boundary from the origin. The model we trained in the previous section did not have a constant offset (called a bias) in the model – computing wT x rather than wT x + b. A simple way to include this in our model is to add an new column to X that has all ones in it. This way, the first weight in our weight vector will always be multiplied by 1 and added.

Below are the results:

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/250a260b74ed7b42b934dbd283f6503a75ba0edc/re.JPG" alt="Image"></kbd>

The code now also produces a plot showing the negative log-likelihood for the bias and no-bias models over the course of training. If we change the learning rate (also called the step size), we could see significant differences in how this plot behaves – and in our accuracies.
Below are a few example resulting models:

Step Size = 0.00001
Unbiased Train accuracy: 85.62%
Biased Train accuracy: 90.77%

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/d0e5da631ffc6d45067106655a46be86f371239a/0.00001.JPG" alt="Image" width="500"></kbd>

Step Size = 0.0001
Unbiased Train accuracy: 86.27%
Biased Train accuracy: 96.35%

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/bf733f03e7fbcb08188828caef331a7d1ef0b191/0.0001.JPG" alt="Image" width="500"></kbd>

Step Size = 0.1
Unbiased Train accuracy: 83.69%
Biased Train accuracy: 95.28%

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/9a16f2639a00b47d1d5bd7f3c53347ed3e9b19e0/0.1.JPG" alt="Image" width="500"></kbd>

To note: a step size that is too large, such as 0.1, will always miss the optimal value and fail to converage, which explains the rather unpleasant-looking graph.

After printing out K-fold cross validation results (mean and standard deviation of accuracy) for K = 2, 3, 4, 5, 10, 20, and 50, the code will output predictions for the test set to test_predicted_2.csv. Hyperparameters can be adjusted for higher prediction accuracy.

# Simple Neural Network

The goal is to implement a feed-forward neural network model for predicting the value of a drawn digit. We are using a subset of the MNIST dataset commonly used in machine learning research papers. A few example of these handwritten-then-digitized digits from the dataset are shown below:

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/ee81f31108a8a612b8c7a4f9f5fef8d1351a7648/ft4.JPG" width="500"></kbd>

Each digit is a 28 × 28 greyscale image with values ranging from 0 to 256. We represent an image as a row vector x ∈ R 1×784 where the image has been serialized into one long vector. Each digit has an associated class label from 0,1,2,...,9 corresponding to its value. We provide three dataset splits for this homework – a training set containing 5000 examples, a validation set containing 1000, and our test set containing 4220 (no labels).

Unlike the previous classification tasks we’ve examined, we have 10 different possible class labels here. How do we measure error of our model? Let’s formalize this a little and say we have a dataset D = {xi , yi} N i=1 with yi ∈ {0, 1, 2, ..., 9}. Assume we have a model f(x; θ) parameterized by a set of parameters θ that predicts P(Y |X = x) (a distribution over our labels given an input). Let’s refer to P(Y = c|X = x) predicted from this model as pc|x for compactness. We can write this output as a categorical distribution:

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/73979ebe8de8271a186b4ae236bdbb15c57d2e27/tgr5.JPG" width="500"></kbd>

where I[condition] is the indicator function that is 1 if the condition is true and 0 otherwise. Using this, we can write
our negative log-likelihood of a single example as as:

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/540b910c1174d3cbab42230b5263cab5df6f5e9d/fewer.JPG" width="500"></kbd>

We’ll consider feed-forward neural networks composed of a sequence of linear layers xW1 + b1 and non-linear activation functions g1(·). As such, a network with 3 of these layers stacked together can be written:

b3 + g2(b2 + g1(b1 + x ∗ W1) ∗ W2) ∗ W3

Considering this simple 3-layer neural network, there are quite a few parameters spread out through the function – weight matrices W3, W2, W1 and biases vectors b3, b2, b1. Suppose we would like to find parameters that minimize our loss L that measures our error in the network’s prediction.
How can we update the weights to reduce this error? Let’s use gradient descent and start by writing out the chain rule for the gradient of each of these. I’ll work backwards from W3 to W1 to expose some structure here.

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/37d858fbe66e47e56431be95964ac907c1c6415c/42.JPG" width="500"></kbd>

As highlighted in color above, we end up reusing the same intermediate terms over and over as we compute derivatives for weights further and further from the output in our network. This suggests the straight-forward backpropagation algorithm for computing these efficiently. Specifically, we will compute these intermediate colored terms starting from the output and working backwards.

One convenient way to implement backpropagation is to consider each layer (or operation) f as having a forward pass that computes the function output normally as:

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/8a6f987731c2f011b89b449cabce3d2262868f8c/434343.JPG" width="500"></kbd>

and a backward pass that takes in the gradient up to this point in our backward pass and then outputs the gradient of the loss with respect to its input:

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/f3b479af990dd4d79f46cf6afc5199c3b278f9c9/5455454.JPG" width="500"></kbd>

The backward operator will also compute any gradients with respect to parameters of f and store them to be used in a gradient descent update step after the backwards pass.

we’ll implement the backward pass of a linear layer. To do so, we’ll need to be able to compute dZ/db, dZ/dW, and dZ/dX. For each, we’ll start by considering the problem for a single training example x (i.e. a single row of X) and then generalize to the batch setting. In this single-example setting, z = xW + b such that z, b ∈ R 1×c , x ∈ R 1×d , and W ∈ R d×c . Once we solve this case, extending to the batch setting just requires summing over the gradient terms for each example.

This model is trained on the training set and evaluated once per epoch on the validation data. After training, it produces a plot of results below. This curve plots training and validation loss (cross-entropy in this case) over training iterations (in red and measured on the left vertical axis). It also plots training and validation accuracy (in blue and measures on the right vertical axis). As you can see, this model achieves between 80% and 90% accuracy on the validation set.

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/2b4f7210332fb883a7b1ca6bf2dc157b0e584085/4324.JPG" width="600"></kbd>

Neural networks have many hyperparameters. These range from architectural choices (How many layers? How wide should each layer be? What activation function should be used? ) to optimization parameters (What batch size for stochastic gradient descent? What step size (aka learning rate)? How many epochs should I train? ). 

Optimization parameters in Stochastic Gradient Descent are very inter-related. Large batch sizes mean less noisy estimates of the gradient, so larger step sizes could be used. But larger batch sizes also mean fewer gradient updates per epoch, so we might need to increase the max epochs. Getting a good set of parameters that work well can be tricky and requires checking the validation set performance. Further, these “good parameters” will vary model-to-model.

Model for Step size: 0.0001

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/ed3cef3727eb78bba41286f30862c05d07a79380/0.00012.JPG" width="500"></kbd>

Model for Step size: 5

<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/93283c9b6bf8c91dca07515112abb12d49c7403c/5.JPG" width="500"></kbd>

As networks get deeper (or have more layers) they tend to become able to fit more complex functions (though this also may lead to overfitting). However, this also means the backpropagated gradient has many product terms before reaching lower levels – resulting in the magnitude of the gradients being relatively small. This has the effect of making learning slower. Certain activation functions make this better or worse depending on the shape of their derivative. One popular choice is to use a Rectified Linear Unit or ReLU activation that computes:

ReLU(x) = max(0, x)

Default Sigmoid Model:
<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/2b4f7210332fb883a7b1ca6bf2dc157b0e584085/4324.JPG" width="500"></kbd>

ReLU Activation Function Model:
<kbd><img src="https://github.com/FluffyCrocodile/Storage/blob/477ade7119b5d7844e9f469483051ab170e870fe/relu.JPG" width="500"></kbd>

# k-Means Clustering

