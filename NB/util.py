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
    parser = optparse.OptionParser(description='run Naive Bayes')

    parser.add_option('-r', '--train_filename', type='string', help='path to' +\
        ' train arff file')
    parser.add_option('-e', '--test_filename', type='string', help='path to' +\
        ' test arff file')

    (opts, args) = parser.parse_args()

    mandatories = ['train_filename', 'test_filename']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts
