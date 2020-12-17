"""
Main file where partitions are created and there is a call to the NaiveBayes class.
Also computing accuracy and printing it as well as the confusion matrix.

Author: Faryal Khan
"""
# import util
# from NaiveBayes import *

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def main():
    """
    Main function

    """
    df = pd.read_csv("headlines_train.csv")
    model = GaussianNB()


    X = df.iloc[:, :-1]
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =40)

    y_pred = model.fit(X_train, y_train).predict(X_test)

    print("Accuracy: " + str( (X_test.shape[0]- (y_test != y_pred).sum() )/ X_test.shape[0]) + " " + str((X_test.shape[0]- (y_test != y_pred).sum() )) + " out of " + str(
        X_test.shape[0]) + " correct.")

    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))



main()