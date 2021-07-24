import cv2
import numpy as np
import pandas as pd
import sklearn.model_selection as sml
import math
from numpy import ndarray, exp, pi, sqrt

data = pd.read_csv('breast_cancer.csv', header=None)
x, y = np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])
train_x, test_x, train_y, test_y = sml.train_test_split(x, y, test_size=0.2, random_state=0)

avg=np.zeros((train_x.shape[1],2),dtype=np.double)
var=np.zeros((train_x.shape[1],2),dtype=np.double)




count_0 = 0
count_1 = 0
for i in train_y:
    if i == 0:
        count_0 += 1
    else:
        count_1 += 1
prior = np.array([count_0 / len(train_y), count_1 / len(train_y)])



for i in range(train_x.shape[1]):
    temp_0 = np.zeros(train_x.shape[1], dtype=np.double)
    temp_1 = np.zeros(train_x.shape[1], dtype=np.double)
    for j in range(train_x.shape[0]):
        if train_y[j] == 0:
            temp_0 = np.append(temp_0,train_x[j][i])
        elif train_y[j] == 1:
            temp_1 = np.append(temp_1,train_x[j][i])
    var[i][0] = temp_0.var()
    var[i][1] = temp_1.var()
    avg[i][0] = temp_0.mean()
    avg[i][1] = temp_1.mean()


def GetLikelihood(x):
   pc_0 = np.ones(x.shape[0], dtype=np.double)
   pc_1 = np.ones(x.shape[0], dtype=np.double)
   for i in range(x.shape[0]):
       for j in range(x.shape[1]):
           pc_0[i] *= 1 / (pow(2 * math.pi, 0.5) * var[j][0]) * math.exp(-(pow(x[i][j] - avg[j][0], 2) / (2 * var[j][0])))
           pc_1[i] *= 1 / (pow(2 * math.pi, 0.5) * var[j][1]) * math.exp(-(pow(x[i][j] - avg[j][1], 2) / (2 * var[j][1])))
   return pc_0,pc_1
   pass

result=[]
for i in range(len(test_x)):
    if GetLikelihood(test_x)[0][i] * prior[0] > GetLikelihood(test_x)[1][i] * prior[1]:
        result.append(0)
    else:
        result.append(1)
#likelihood = np.apply_along_axis(GetLikelihood, axis=1, arr=test_x)
print(result)
correct = np.sum(result == test_y).astype(float)
print("Accuracy:", correct / len(test_y))
