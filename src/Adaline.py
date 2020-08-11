import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import random
import time
from getCategories import Category


def train(X, y, W, learning_rate, epochs):
    costs = []

    for iteration in range(epochs):
        outputs = np.dot(X, W[1:]) + W[0] #calculate the output from the weight and the input we have includes the bias
        errors = (y - outputs) #the error between the output and the true value
        W[1:] += learning_rate * np.dot(X.T, errors) #compute the new weigh
        W[0] += learning_rate * errors.sum() #add to the bias (compute)
        cost = (errors ** 2).sum() / 2.0 #compute the loss
        costs.append(cost) #add the loss so we can print it like graph

    return W, np.array(costs)


def predict(x, W): #this function calculate the prediction of out model as per the compte weights and bias we got
    return np.where(np.dot(x, W[1:]) + W[0] >= 0., 1, -1)


def actual_val(y):
    '''
    we compute the number of the actual values for the label
    '''
    pos = neg = 0
    for item in y:
        if item == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg


def check_predict(y, predict):
    '''
    check how many values we predict as well
    to compute the true and negative values
    '''
    TP = FN = FP = TN = 0
    for i in range(0, len(y)):
        if predict[i] == 1:
            if y[i] == 1:
                TP += 1
            elif y[i] == -1:
                FN += 1

        elif predict[i] == -1:
            if y[i] == 1:
                FP += 1
            elif y[i] == -1:
                TN += 1
    return TP, FN, FP, TN


def checkScore(TP, FN, FP, TN):
    '''
    compute thepercents of the values we got
    to compute the recall, precision and the accuracy
    '''
    print("true positive: ", TP)
    print("false negative: ", FN)
    print("false positive: ", FP)
    print("true negative: ", TN)

    total_all = TP + FN + FP + TN
    total_true = TP + TN
    acc = total_true / total_all
    total_pos = TP + FP
    try:
        precision = TP / total_pos
        recall = TP / (true_pos + FN)
        print("accuracy: ", round(acc, 3))
        print("precision: ", round(precision, 3))
        print("recall: ", round(recall, 3))
    except ZeroDivisionError:
        print("inf")

mnist = np.load("../Dataset/DataSet.npy", allow_pickle=True)
random.shuffle(mnist)

x_train = np.array([mnist[i].features for i in range(0, int(len(mnist) * 0.8))]).astype(np.float64)
x_test = np.array([mnist[i].features for i in range(int(len(mnist) * 0.8), len(mnist))]).astype(np.float64)
y_train = np.array([mnist[i].label for i in range(0, int(len(mnist) * 0.8))])
y_test = np.array([mnist[i].label for i in range(int(len(mnist) * 0.8), len(mnist))])

y = []

for i in y_train:
    if i[0] == 1:
        y.append(1)
    else:
        y.append(-1)

y_train = np.array(y).astype(np.float64)

y = []

for i in y_test:
    if i[0] == 0:
        y.append(-1)
    elif i[0] == 1:
        y.append(1)

y_test = np.array(y).astype(np.float64)

start = time.time() #the start time of the adaline algorith without cross validation

x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis = 0)
x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis = 0)

learning_rate = 0.001
epochs = 10000

random_gen = np.random.RandomState(1) #to get random weights values at start
weights = random_gen.normal(loc = 0.0, scale = 0.01, size = x_train.shape[1] + 1) #+bias
weights, costs = train(x_train, y_train, weights, learning_rate, epochs) #train the model of adaline

plt.plot(range(1, len(costs) + 1), costs, color='red')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()

predictions = predict(x_test, weights) #the values we predict with the values of our weights and biases
actual_recurred, actual_not_recurred = actual_val(y_test)
(true_pos, false_neg, false_pos, true_neg) = check_predict(y_test, predictions)

checkScore(true_pos, false_neg, false_pos, true_neg)

print("-------------------------------\nFirst model: 0.7 train 0.3 test\n-------------------------------")
print("true positive: ", round(true_pos / actual_recurred, 3))
print("false negative: ", round(false_neg / actual_recurred, 3))
print("true negative: ", round(true_neg / actual_not_recurred, 3))
print("false positive: ", round(false_pos / actual_not_recurred, 3))
print("\ntime: ", round(time.time() - start, 3), 'seconds')
