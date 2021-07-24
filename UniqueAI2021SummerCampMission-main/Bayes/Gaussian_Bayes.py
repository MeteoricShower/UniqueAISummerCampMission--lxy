# -*- coding=utf-8 -*-
# @Time :2021/6/17 11:45
# @Author :LiZeKai
# @Site : 
# @File : Gaussian_Bayes.py
# @Software : PyCharm

"""
    对于连续性数据, 使用GaussBayes
    以乳腺癌数据集为例
"""
from collections import Counter
import numpy as np
import pandas as pd
import sklearn.model_selection as sml
import math
from numpy import ndarray, exp, pi, sqrt


class GaussBayes:

    def __init__(self):
        self.prior = None
        self.var = None
        self.avg = None
        self.likelihood = None
        self.tag_num = None

    # calculate the prior probability of p_c
    def GetPrior(self, label):
        count_0=0
        count_1=0
        for i in label:
            if i==0:
                count_0+=1
            else:
                count_1+=1

        self.prior = [count_0 / len(label), count_1 / len(label)]
        pass
        
    # calculate the average
    def GetAverage(self, data, label):
        self.avg=np.zeros((data.shape[1],2),dtype=np.double)
        for i in range(data.shape[1]):
            sum_0=0
            sum_1=0
            count_0 = 0
            count_1 = 0
            for j in range(data.shape[0]):
                if label[j]==0:
                    sum_0+=data[j][i]
                    count_0 += 1
                elif label[j]==1:
                    sum_1+=data[j][i]
                    count_1 += 1
            self.avg[i][0]=sum_0/count_0
            self.avg[i][1]=sum_1/count_1


        pass

    # calculate the std
    def GetStd(self, data, label):
        self.var = np.zeros((data.shape[1], 2), dtype=np.double)
        for i in range(data.shape[1]):
            temp_0 = np.zeros(data.shape[1],dtype=np.double)
            temp_1 = np.zeros(data.shape[1], dtype=np.double)
            for j in range(data.shape[0]):
                if label[j]==0:
                    temp_0 = np.append(temp_0,data[j][i])
                elif label[j]==1:
                    temp_1 = np.append(temp_1,data[j][i])
            self.var[i][0]=temp_0.var()
            self.var[i][1]=temp_1.var()
        pass

    # calculate the likelihood based on the density function
    def GetLikelihood(self, x):
        pc_0 = np.ones(x.shape[0], dtype=np.double)
        pc_1 = np.ones(x.shape[0], dtype=np.double)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                pc_0[i] *= 1 / (pow(2 * math.pi, 0.5) * self.var[j][0]) * math.exp(
                    -(pow(x[i][j] - self.avg[j][0], 2) / (2 * self.var[j][0])))
                pc_1[i] *= 1 / (pow(2 * math.pi, 0.5) * self.var[j][1]) * math.exp(
                    -(pow(x[i][j] - self.avg[j][1], 2) / (2 * self.var[j][1])))
        return pc_0, pc_1
        pass

    def fit(self, data, label):
        self.tag_num = len(np.unique(label))
        self.GetPrior(label)
        self.GetAverage(data, label)
        self.GetStd(data, label)

    def predict(self, data):
        result = []
        for i in range(len(test_x)):
            if self.GetLikelihood(data)[0][i] * self.prior[0] > self.GetLikelihood(data)[1][i] * self.prior[1]:
                result.append(0)
            else:
                result.append(1)
        return result


if __name__ == '__main__':
    data = pd.read_csv('breast_cancer.csv', header=None)
    x, y = np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])
    train_x, test_x, train_y, test_y = sml.train_test_split(x, y, test_size=0.2, random_state=0)
    model = GaussBayes()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    correct = np.sum(pred_y == test_y).astype(float)
    print("Accuracy:", correct / len(test_y))
