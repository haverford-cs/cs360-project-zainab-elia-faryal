"""
Authors: Zainab Batool
Date: 10/22/2020
Description: A class to compute logistic regression solution
"""
import util
import numpy as np
from LogisticRegression import *


def main():
    total=0
    correct=0
    confusion_matrix=np.zeros((2,2))
    opts = util.parse_args()
    #read train and test files
    train_partition = util.read_csv(opts.train_filename)
    test_partition  = util.read_csv(opts.test_filename)
    #create model and train it
    model = LogisticRegression()
    model.fit_SGD(train_partition.X,train_partition.y,opts.alpha)

    n=test_partition.y.shape[0]
    X = model.generate_logistic_features(test_partition.X)
    #iterate through each test example, make prediction then check and compute accuracy
    for i in range(0,n):
        y_pred=model.predict(X[i])
        if y_pred>=0.5:
            y_pred=1
        else:
            y_pred=0
        #check accuracy and fill in matrix
        y = int(test_partition.y[i])
        if check_accuracy(y_pred,test_partition.y[i]):
            confusion_matrix[y,y]+=1
            correct+=1
        else:
            confusion_matrix[y,y_pred]+=1
        total+=1
    accuracy=calculate_accuracy(correct,total)
    print(f"{correct} out of {total} correct\n accuracy = {accuracy}")
    print("Prediction")
    print(confusion_matrix)

def check_accuracy(num1,num2):
    """Check if our label matches real label"""
    return num1==num2

def calculate_accuracy(correct, total):
    """Calculate accuracy of the nb_model"""
    return (correct/total)

if __name__ == "__main__" :
    main()
