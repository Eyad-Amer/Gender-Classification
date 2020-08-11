from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from matplotlib.colors import ListedColormap
import random
import time
from sklearn import metrics
from getCategories import Category

mnist = np.load("../Dataset/DataSet.npy", allow_pickle=True)
random.shuffle(mnist)

x_train = np.array([mnist[i].features for i in range(0, int(len(mnist) * 0.8))]).astype(np.float64)
x_test = np.array([mnist[i].features for i in range(int(len(mnist) * 0.8), len(mnist))]).astype(np.float64)
y_train = np.array([mnist[i].label for i in range(0, int(len(mnist) * 0.8))])
y_test = np.array([mnist[i].label for i in range(int(len(mnist) * 0.8), len(mnist))])

y = []

for i in y_train:
    if i[0] == 0:
        y.append(-1)
    elif i[0] == 1:
        y.append(1)

y_train = np.array(y).astype(np.float64)

y = []

for i in y_test:
    if i[0] == 0:
        y.append(-1)
    elif i[0] == 1:
        y.append(1)

y_test = np.array(y).astype(np.float64)


clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(x_train, y_train)
AdaBoostClassifier(n_estimators=100, random_state=0)
predict = clf.predict(x_test)


print("accuracy", metrics.accuracy_score(y_test, predict))