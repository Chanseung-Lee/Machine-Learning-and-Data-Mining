import numpy as np
import warnings

np.random.seed(42)
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
step_size = 0.0001
max_iters = 1000


def main():
    # Load the training data
    logging.info("Loading data")
    X_train, y_train, X_test = loadData()

    logging.info("\n---------------------------------------------------------------------------\n")

    # Fit a logistic regression model on train and plot its losses
    logging.info("Training logistic regression model (No Bias Term)")
    w, losses = trainLogistic(X_train, y_train)
    y_pred_train = X_train @ w >= 0


    logging.info("Learned weight vector: {}".format([np.round(a, 4)[0] for a in w]))
    logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train) * 100))

    logging.info("\n---------------------------------------------------------------------------\n")

    X_train_bias = dummyAugment(X_train)

    # Fit a logistic regression model on train and plot its losses
    logging.info("Training logistic regression model (Added Bias Term)")
    w, bias_losses = trainLogistic(X_train_bias, y_train)
    y_pred_train = X_train_bias @ w >= 0

    logging.info("Learned weight vector: {}".format([np.round(a, 4)[0] for a in w]))
    logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train) * 100))

    plt.figure(figsize=(16, 9))
    plt.plot(range(len(losses)), losses, label="No Bias Term Added")
    plt.plot(range(len(bias_losses)), bias_losses, label="Bias Term Added")
    plt.title("Logistic Regression Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Likelihood")
    plt.legend()
    plt.show()

    logging.info("\n---------------------------------------------------------------------------\n")

    logging.info("Running cross-fold validation for bias case:")

    for k in [2, 3, 4, 5, 10, 20, 50]:
        cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k)
        logging.info(
            "{}-fold Cross Val Accuracy -- Mean (stdev): {:.4}% ({:.4}%)".format(k, cv_acc * 100, cv_std * 100))


    ####################################################
    # Write the code to make your test submission here
    ####################################################

    #pred_test_y = predict(train_X, train_y, test_X, best_k)

    # add index and header then save to file
    #test_out = np.concatenate((np.expand_dims(np.array(range(2000), dtype=int), axis=1), pred_test_y), axis=1)
    #trained_weight, losses = trainLogistic()

    wei, losses = trainLogistic(X_train_bias, y_train)
    Biastest = dummyAugment(X_test)
    y_pred_test = Biastest @ wei >= 0
    test_out = np.concatenate((np.expand_dims(np.array(range(233), dtype=int), axis=1), y_pred_test), axis=1)
    header = np.array([["id", "type"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted_hw2.csv', test_out, fmt='%s', delimiter=',')


######################################################################
# logistic
######################################################################
# Given an input vector z, return a vector of the outputs of a logistic
# function applied to each input value
#
# Input:
#   z --   a n-by-1 vector
#
# Output:
#   logit_z --  a n-by-1 vector where logit_z[i] is the result of
#               applying the logistic function to z[i]
######################################################################
def logistic(z):

    logit_z = np.zeros((len(z), 1))
    #print("what came into logistic:")
    #print(z)
    #print("logit_z:")
    #print(logit_z)
    warnings.filterwarnings('ignore')

    for i in range(len(z)):
        a = z[i]
        b = 1 / (1 + np.exp(-a))
        logit_z[i,0] = b

    #print("after logi, logit_z:")
    #print(logit_z)

    return logit_z


######################################################################
# calculateNegativeLogLikelihood
######################################################################
# Given an input data matrix X, label vector y, and weight vector w
# compute the negative log likelihood of a logistic regression model
# using w on the data defined by X and y
#
# Input:
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   y --    a n-by-1 vector representing the labels of the examples in X
#
#   w --    a d-by-1 weight bector
#
# Output:
#   nll --  the value of the negative log-likelihood
######################################################################
def calculateNegativeLogLikelihood(X, y, w):

    total = 0

    for i in range(len(y)):
        a = np.log(logistic(w.T@X[i]) + 0.000001)*y[i]
        b = (1-y[i])*(np.log(1-logistic(w.T@X[i]) + 0.0000001))
        c = a+b
        total = total + c

    nll = -total
    #print("nll: ", nll)
    return nll[0,0]


######################################################################
# trainLogistic
######################################################################
# Given an input data matrix X, label vector y, maximum number of
# iterations max_iters, and step size step_size -- run max_iters of
# gradient descent with a step size of step_size to optimize a weight
# vector that minimizies negative log-likelihood on the data defined
# by X and y
#
# Input:
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   y --    a n-by-1 vector representing the labels of the examples in X
#
#   max_iters --   the maximum number of gradient descent iterations
#
#   step_size -- the step size (or learning rate) for gradient descent
#
# Output:
#   w --  the d-by-1 weight vector at the end of training
#
#   losses -- a list of negative log-likelihood values for each iteration
######################################################################

def trainLogistic(X, y, max_iters=max_iters, step_size=step_size):
    
    # Initialize our weights with zeros
    w = np.zeros((X.shape[1], 1))
    w_grad = np.zeros((X.shape[1], 1))

    #print("starting w:")
    #print(w)
    # Keep track of losses for plotting
    losses = [calculateNegativeLogLikelihood(X, y, w)]

    # Take up to max_iters steps of gradient descent
    for i in range(max_iters):
        total = np.zeros((X.shape[1], 1))
        total = X.T @ (logistic(X @ w) - y)
        '''
        for p in range(len(X)):
            please = logistic(X[p]@w)-y[p]
            bro = np.array(X[p])[np.newaxis]
            new = bro.T
            total = total + new@please
            
        '''

        w_grad = total


        # This is here to make sure your gradient is the right shape
        assert (w_grad.shape == (X.shape[1], 1))

        # Take the update step in gradient descent
        w = w - step_size * w_grad

        # Calculate the negative log-likelihood with the
        # new weight vector and store it for plotting later
        losses.append(calculateNegativeLogLikelihood(X, y, w))


    return w, losses


######################################################################
# dummyAugment
######################################################################
# Given an input data matrix X, add a column of ones to the left-hand
# side
#
# Input:
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
# Output:
#   aug_X --  a n-by-(d+1) matrix of examples where each row
#                   corresponds to a single d-dimensional example
#                   where the the first column is all ones
#
######################################################################
def dummyAugment(X):

    singles = np.ones((len(X), 1))
    aug_X = np.hstack((singles, X))

    return aug_X


# Given a matrix X (n x d) and y (n x 1), perform k fold cross val.
def kFoldCrossVal(X, y, k):
    fold_size = int(np.ceil(len(X) / k))

    rand_inds = np.random.permutation(len(X))
    X = X[rand_inds]
    y = y[rand_inds]

    acc = []
    inds = np.arange(len(X))
    for j in range(k):

        start = min(len(X), fold_size * j)
        end = min(len(X), fold_size * (j + 1))
        test_idx = np.arange(start, end)
        train_idx = np.concatenate([np.arange(0, start), np.arange(end, len(X))])
        if len(test_idx) < 2:
            break

        X_fold_test = X[test_idx]
        y_fold_test = y[test_idx]

        X_fold_train = X[train_idx]
        y_fold_train = y[train_idx]

        w, losses = trainLogistic(X_fold_train, y_fold_train)

        acc.append(np.mean((X_fold_test @ w >= 0) == y_fold_test))

    return np.mean(acc), np.std(acc)


# Loads the train and test splits, passes back x/y for train and just x for test
def loadData():
    train = np.loadtxt("train_cancer.csv", delimiter=",")
    test = np.loadtxt("test_cancer_pub.csv", delimiter=",")

    X_train = train[:, 0:-1]
    y_train = train[:, -1]
    X_test = test

    return X_train, y_train[:,
                    np.newaxis], X_test  # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.


main()
