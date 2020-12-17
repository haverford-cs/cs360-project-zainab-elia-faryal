"""
Authors: Zainab Batool
Date: 10/22/2020
Description: A class to compute logistic regression solution
"""
import numpy as np
import math

class LogisticRegression:
    def __init__(self, reg_param=0) :
        """
        Logistic regression.
        weights (numpy array of shape (p+1,)) -- estimated coefficients for the
            logistic regression problem (these are the w's from in class)
        lambda_ (float) -- regularization parameter
        """
        self.weights = None
        self.lambda_ = reg_param

    def generate_logistic_features(self, X) :
        """
        params: X (numpy array of shape (n,p)) -- features
        Adds fake 1 feature to every example in X to return (n,p+1) matrix
        """
        n,p = X.shape
        X0 = np.ones((n,1))
        X= np.hstack((X0,X))
        return X

    def fit_SGD(self, X, y, alpha, eps=1e-4, tmax=2734):

        if self.lambda_ != 0 :
            raise Exception("SGD with regularization not implemented")

        X_copy = X
        X = self.generate_logistic_features(X) # map features
        n,p = X.shape
        self.weights = np.zeros(p)                 # weights
        cost_list  = np.zeros((tmax,1))           # cost per iteration

        # SGD loop
        for t in range(tmax):
            #implement shuffling
            s=np.arange(X.shape[0])
            np.random.shuffle(s)
            X=X[s]
            y=y[s]
            # iterate through examples while keeping track of cost per iteration
            total_cost_term=1
            for i in range(n) :
                #updates weights
                hw=self.predict(X[i])
                change=alpha*(hw-y[i])
                finalchange=change*X[i]
                self.weights-=finalchange
                #update cost
                total_cost_term=total_cost_term*self.single_cost_term(X[i],y[i])
                pass

            #take negative log of total cost term to compute cost of iteration
            cost=-1*math.log(total_cost_term)
            cost_list[t] = cost
            # stop when change in cost is less than hyperparameter epsilon
            if t > 0 and abs(cost_list[t] - cost_list[t-1]) < eps :
                break

    def predict(self, X) :
        """
        Predict output for X.
        Parameters:
            X       -- numpy array of shape (n,p), features
        Returns:
            y       -- numpy array of shape (n,), predictions
        """
        if self.weights is None :
            raise Exception("Model not initialized. Perform a fit first.")

        dotProduct = np.dot(X,self.weights)
        exponent=math.exp(-dotProduct)
        y_pred=1/(1+exponent)

        return y_pred

    def single_cost_term(self,X,y):
        """
        compute the cost term for a single i
        """
        power1=np.power(X,y)
        first_term=self.predict(power1)
        second_term=(1-self.predict(X))**(1-y)
        cost_term=first_term*second_term
        return cost_term
