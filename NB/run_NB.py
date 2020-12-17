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
import util

def main():
    """
    Main function

    """
    opts = util.parse_args()
    df1 = pd.read_csv(opts.train_filename)
    df2 = pd.read_csv(opts.test_filename)
    model = GaussianNB()

    # print(df1.shape)
    # print(df2.shape)

    X_train = df1.iloc[:, :-1]
    y_train = df1[df1.columns[-1]]

    X_test = df2.iloc[:, :-1]
    y_test = df2[df2.columns[-1]]

    y_pred = model.fit(X_train, y_train).predict(X_test)

    print("Accuracy: " + str( (X_test.shape[0]- (y_test != y_pred).sum())/ X_test.shape[0]) + " " + str((X_test.shape[0]- (y_test != y_pred).sum() )) + " out of " + str(
        X_test.shape[0]) + " correct.")

    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))



main()