import statistics

import numpy as np
from numpy import sin, cos, arctan2, sqrt, cross, pi, argmin
import time


def main():
    #############################################################
    # These first bits are just to help you develop your code
    # and have expected ouputs given. All asserts should pass.
    ############################################################

    # I made up some random 3-dimensional data and some labels for us
    example_train_x = np.array([[1, 0, 2], [3, -2, 4], [5, -2, 4],
                                [4, 2, 1.5], [3.2, np.pi, 2], [-5, 0, 1]])
    example_train_xx = np.array([[1, 99], [3, -2], [5, 1],
                                [4, 2]])
    example_train_y = np.array([[0], [1], [1], [1], [0], [1]])
    example_train_yy = np.array([[0], [1], [1], [1]])

    #########
    # Sanity Check 1: If I query with examples from the training set
    # and k=1, each point should be its own nearest neighbor

    for i in range(len(example_train_x)):
        assert ([i] == get_nearest_neighbors(example_train_x, example_train_x[i], 1))

    #########
    # Sanity Check 2: See if neighbors are right for some examples (ignoring order)
    nn_idx = get_nearest_neighbors(example_train_x, np.array([1, 4, 2]), 2)
    assert (set(nn_idx).difference(set([4, 3])) == set())

    nn_idx = get_nearest_neighbors(example_train_x, np.array([1, -4, 2]), 3)
    assert (set(nn_idx).difference(set([1, 0, 2])) == set())

    nn_idx = get_nearest_neighbors(example_train_x, np.array([10, 40, 20]), 5)
    assert (set(nn_idx).difference(set([4, 3, 0, 2, 1])) == set())

    #########
    # Sanity Check 3: Neighbors for increasing k should be subsets
    query = np.array([10, 40, 20])
    p_nn_idx = get_nearest_neighbors(example_train_x, query, 1)
    for k in range(2, 7):
        nn_idx = get_nearest_neighbors(example_train_x, query, k)
        assert (set(p_nn_idx).issubset(nn_idx))
        p_nn_idx = nn_idx

    #########
    # Test out our prediction code
    queries = np.array([[10, 40, 20], [-2, 0, 5], [0, 0, 0]])
    pred = predict(example_train_x, example_train_y, queries, 3)
    assert (np.all(pred == np.array([[0], [1], [0]])))

    print(cross_validation(example_train_xx, example_train_yy, 2, 1))
    #########
    # Test our our accuracy code
    true_y = np.array([[0], [1], [2], [1], [1], [0]])
    pred_y = np.array([[5], [1], [0], [0], [1], [0]])
    assert (compute_accuracy(true_y, pred_y) == 3 / 6)

    pred_y = np.array([[5], [1], [2], [0], [1], [0]])
    assert (compute_accuracy(true_y, pred_y) == 4 / 6)

    #######################################
    # Now on to the real data!
    #######################################

    # Load training and test data as numpy matrices
    train_X, train_y, test_X = load_data()

    #######################################
    # Q9 Hyperparmeter Search
    #######################################

    # Search over possible settings of k

    print("Performing 4-fold cross validation")
    for k in [1, 3, 5, 7, 9, 99, 999, 8000]:
        t0 = time.time()

        #######################################
        # TODO Compute train accuracy using whole set
        #######################################
        train_acc = 0

        postulated_y = predict(train_X, train_y, train_X, k)
        train_acc = compute_accuracy(train_y, postulated_y)

        #######################################
        # TODO Compute 4-fold cross validation accuracy
        #######################################
        val_acc, val_acc_var = 0, 0

        val_acc, val_acc_var = cross_validation(train_X, train_y, 4, k)

        t1 = time.time()
        print("k = {:5d} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t\t[exe_time = {:.2f}]".format(k,
                                                                                                           train_acc * 100,
                                                                                                           val_acc * 100,
                                                                                                           val_acc_var * 100,
                                                                                                           t1 - t0))

    #######################################

    #######################################
    # Q10 Kaggle Submission
    #######################################

    # TODO set your best k value and then run on the test set

    best_k = 200

    # Make predictions on test set
    pred_test_y = predict(train_X, train_y, test_X, best_k)

    # add index and header then save to file
    test_out = np.concatenate((np.expand_dims(np.array(range(2000), dtype=int), axis=1), pred_test_y), axis=1)
    header = np.array([["id", "income"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')



######################################################################
# get_nearest_neighbors
######################################################################
# Finds and returns the index of the k examples nearest to
# the query point. Here, nearest is defined as having the
# lowest Euclidean distance. This function does the bulk of the
# computation in kNN. As described in the homework, you'll want
# to use efficient computation to get this done. Check out
# the documentaiton for np.linalg.norm (with axis=1) and broadcasting
# in numpy.
#
# Input:
#   example_set --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   idx_of_nearest --   a k-by- list of indices for the nearest k
#                       neighbors of the query point
######################################################################

def get_nearest_neighbors(example_set, query, k):
    # TODO


    diff = example_set - query
    dist = sqrt(np.sum(diff ** 2, axis=-1))

    sorted_dist = np.argsort(dist)
    idx_of_nearest = sorted_dist[0:k]

    return idx_of_nearest


######################################################################
# knn_classify_point
######################################################################
# Runs a kNN classifier on the query point
#
# Input:
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_label --   either 0 or 1 corresponding to the predicted
#                        class of the query based on the neighbors
######################################################################

def knn_classify_point(examples_X, examples_y, query, k):
    # TODO

    neighbor_indices = get_nearest_neighbors(examples_X, query, k)
    zero = 0
    one = 0

    for i in neighbor_indices:
        if examples_y[i] == 0:
            zero += 1
        elif examples_y[i] == 1:
            one += 1
        else:
            raise Exception("invalid class label detected")

    if zero > one:
        predicted_label = 0
    else:
        predicted_label = 1

    return predicted_label



######################################################################
# cross_validation
######################################################################
# Runs K-fold cross validation on our training data.
#
# Input:
#   train_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   train_Y --  a n-by-1 vector of example class labels
#
# Output:
#   avg_val_acc --      the average validation accuracy across the folds
#   var_val_acc --      the variance of validation accuracy across the folds
######################################################################

def cross_validation(train_X, train_y, num_folds=4, kay=1):
    # TODO
    split_X = np.split(train_X, num_folds)
    split_Y = np.split(train_y, num_folds)
    fold_accuracies = list()

    #for each fold:
    for i in range(num_folds):
        train_stack_X = np.empty
        train_stack_Y = np.empty
        test_stack_X = np.empty
        test_stack_Y = np.empty

        train_list_x = list()
        train_list_y = list()
        fold_correct = 0
        fold_total = 0
    #the current i is the test fold, the rest are the training folds
        for j in range(num_folds):
            if(i != j):
                #train_stack_X = np.vstack((train_stack_X, split_X[j]))
                #train_stack_Y = np.vstack((train_stack_Y, split_Y[j]))
                train_list_x.append(split_X[j])
                train_list_y.append(split_Y[j])
            if(i == j):
                test_stack_X = split_X[j]
                test_stack_Y = split_Y[j]

        train_stack_X = np.vstack(train_list_x)
        train_stack_Y = np.vstack(train_list_y)
    #predict the labels of the test set vectors using knn_classify_point
        for k in range(len(test_stack_X)):
            predicted_label = knn_classify_point(train_stack_X, train_stack_Y, test_stack_X[k], kay)
            if predicted_label == test_stack_Y[k]:
                fold_correct += 1

            fold_total += 1

        fold_accuracies.append(fold_correct/fold_total)

    avg_val_acc = Average(fold_accuracies)
    varr_val_acc = statistics.variance(fold_accuracies)

    return avg_val_acc, varr_val_acc


def Average(lst):
    return sum(lst) / len(lst)



######################################################################
# compute_accuracy
######################################################################
# Runs a kNN classifier on the query point
#
# Input:
#   true_y --  a n-by-1 vector where each value corresponds to
#              the true label of an example
#
#   predicted_y --  a n-by-1 vector where each value corresponds
#                to the predicted label of an example
#
# Output:
#   predicted_label --   the fraction of predicted labels that match
#                        the true labels
######################################################################

def compute_accuracy(true_y, predicted_y):
    accuracy = np.mean(true_y == predicted_y)
    return accuracy


######################################################################
# Runs a kNN classifier on every query in a matrix of queries
#
# Input:
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   queries_X --    a m-by-d matrix representing a set of queries
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_y --   a m-by-1 vector of predicted class labels
######################################################################

def predict(examples_X, examples_y, queries_X, k):
    # For each query, run a knn classifier
    predicted_y = [knn_classify_point(examples_X, examples_y, query, k) for query in queries_X]

    return np.array(predicted_y, dtype=int)[:, np.newaxis]


# Load data
def load_data():
    traindata = np.genfromtxt('train.csv', delimiter=',')[1:, 1:]
    train_X = traindata[:, :-1]
    train_y = traindata[:, -1]
    train_y = train_y[:, np.newaxis]

    test_X = np.genfromtxt('test_pub.csv', delimiter=',')[1:, 1:]

    return train_X, train_y, test_X


if __name__ == "__main__":
    main()