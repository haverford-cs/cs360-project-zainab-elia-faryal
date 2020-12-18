"""
File implementing Logistic Regression functions (some functions from Lab 3).
Author: Faryal Khan/Sara Mathieson
Date: 10/20/20
"""
import math
import os
import numpy as np

class Data:

    def __init__(self, X=None, y=None):
        """
        Data class.
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
        """
        # n = number of examples, p = dimensionality
        self.X = X
        self.y = y

    def load(self, filename):
        """
        Load csv file into X array of features and y array of labels.
        filename (string)
        """
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, 'input', filename)

        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]

def load_data(filename):
    data = Data()
    data.load(filename)
    return data


class LogisticRegression:

    def __init__(self, alpha):
        """
        weights_ (numpy array of shape (p+1,)) -- estimated weights for the
            logistic regression problem
        """
        self.weights_ = None
        self.alpha = alpha

    def log_func(self, z):
        return 1/(1+math.exp(-z))

    def generate_features(self, X):
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m] and adds 1.
        params: X (numpy array of shape (n,p)) -- features
        returns: Phi (numpy array of shape (n,1+p*m) -- mapped features
        """
        n, p = X.shape
        Phi = np.insert(X, 0, 1.0, axis=1)
        return Phi

    def predict(self, weights, X):
        """
        Predict output for X.
        Parameters:
            X       -- numpy array of shape (n,p), features
        Returns:
            y       -- numpy array of shape (n,), predictions
        """
        if weights is None:
            raise Exception("Model not initialized properly.")
        X = self.generate_features(X)  # map features
        predictions = []

        l = len(X)
        for i in range(l):
            h = self.log_func(self.weights_ @ X[i])
            predictions.append(h)
        return predictions


    def fit_SGD_helper(self, data, weights):
        z = (weights @ data)
        return self.log_func(z)


    def fit_SGD(self, X, y, eps=1e-4, tmax=100000):
        """
        Finds the coefficients of a polynomial that fits the data using least
        squares stochastic gradient descent.
        Parameters:
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
        """

        alpha = self.alpha

        X = self.generate_features(X) # map features
        n,p = X.shape
        self.weights_ = np.zeros(p)                 #weights
        err_list = np.zeros((tmax,1))           # errors per iteration
        # SGD loop
        for t in range(tmax):
            # iterate through examples
            #np.random.shuffle(X)
            #np.random.shuffle(y)
            predictions = []
            for i in range(len(X)):
                given_y = self.fit_SGD_helper(X[i], self.weights_)
                predictions.append(given_y)
                error = (given_y - y[i]) * X[i]
                self.weights_ = self.weights_ - alpha * error

            costs = self.cost(X, y)
            err_list[t] = costs

            if t > 0 and abs(err_list[t] - err_list[t-1]) < eps:
                break

        print('number of iterations: %d' % (t+1))


    def cost(self, X, y) :
        """
        Calculates the objective function.
        Parameters:
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
        Returns:
            cost    -- float, objective J(b)
        """
        l = len(X)
        cost = 0.0
        for i in range(l):
            h = self.log_func(self.weights_.T @ X[i])
            if h == 1:
                cost = cost + (-y[i] * math.log(h))

            elif h == 0:
                cost = cost + (-(1 - y[i]) * math.log(1 - h))

            else:
                cost = cost + (-y[i] * math.log(h) + -(1 - y[i]) * math.log(1 - h))
        return cost

