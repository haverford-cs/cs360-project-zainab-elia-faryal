"""
Utils for logistic regression.
Author: Zainab batool
Date: 10/22/2020
"""
import math
import numpy as np
import optparse
import sys
import os


def parse_args():
    """Parse command line arguments (train and test csv files)."""
    parser = optparse.OptionParser(description='run logistic regression')

    parser.add_option('-r', '--train_filename', type='string', help='path to' +\
        ' train arff file')
    parser.add_option('-e', '--test_filename', type='string', help='path to' +\
        ' test arff file')
    parser.add_option('-a', '--alpha', type='float', help='alpha')

    (opts, args) = parser.parse_args()

    mandatories = ['train_filename', 'test_filename', 'alpha']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    if opts.alpha < 0:
        print('option "alpha" cannot be negative\n')
        parser.print_help()
        sys.exit()

    return opts

######################################################################
# classes
######################################################################
class Data :

    def __init__(self, X=None, y=None) :
        """
        Data class.
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
        """
        # n = number of examples, p = dimensionality
        self.X = X
        self.y = y

    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.
        filename (string)
        """
        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, filename)

        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]

# Wrapper function around data class
def read_csv(filename) :
    data = Data()
    data.load(filename)
    return data
