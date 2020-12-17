"""
File with main function.
Author: Faryal Khan
Date: 10/20/20
"""
from LogisticRegression import *
import util


def main():

    # load data

    opts = util.parse_args()

    train_data = load_data('headlines_train.csv')
    test_data = load_data('headlines_test.csv')



    alpha = opts.alpha

    test_x = []
    test_y = []
    train_x = []
    train_y = []
    for x_entry in train_data.X:
        train_x.append(x_entry)

    for y_entry in train_data.y:
        train_y.append(y_entry)

    for x_entry in test_data.X:
        test_x.append(x_entry)

    for y_entry in test_data.y:
        test_y.append(y_entry)

    test_X = np.array(test_x)
    test_Y = np.array([test_y])
    train_X = np.array([train_x]).reshape((590606, 1))
    train_Y = np.array([train_y]).reshape((394,))

    model = LogisticRegression(alpha)

    model.fit_SGD(train_data.X, train_data.y)
    weights = model.weights_

    model_predictions = (model.predict(weights, test_X))
    predictions = []
    for value in model_predictions:
        prediction = 0.0
        if value > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    incorrect = 0
    correct = 0
    for i in range(0, len(predictions)):
        if test_data.y[i] != predictions[i]:
            incorrect += 1
        else:
            correct += 1

    print("Accuracy: " + str(correct/(correct+incorrect)) + " " +str(correct) + " out of " + str(correct+incorrect)  + " correct.")


    result = np.zeros((2, 2))

    for i in range(len(test_data.y)):
        result[int(test_data.y[i])][int(predictions[i])] += 1

    print(result)




if __name__ == "__main__" :
    main()
