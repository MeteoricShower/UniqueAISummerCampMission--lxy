# -*- coding=utf-8 -*-
# @Time :2021/6/17 11:47
# @Author :LiZeKai
# @Site : 
# @File : Naive_Bayes.py
# @Software : PyCharm

"""
    python3实现朴素贝叶斯分类器
    以过滤spam为例, 实现二分类器
"""

import numpy as np


class NaiveBayes:

    def __init__(self):
        self.likelihood_1 = None
        self.likelihood_0 = None
        self.p_c_0 = None
        self.p_c_1 = None
        self.tag_num = None

    def fit(self, dataset, labels):
        """
        :param dataset: dataset is an one-hot-encoding numpy array
        :param labels: corresponding tags
        :return: None
        """
        spam=0
        normal=0
        sentence_num,voca_num = dataset.shape
        self.tag_num=voca_num
        for i in labels:
            if i==1:
                spam += 1
            else:
                normal += 1
        self.likelihood_1 = (spam+1)/(len(labels)+2.0)
        self.likelihood_0 = (normal+1)/(len(labels)+2.0)

        num_word = np.zeros((2, voca_num), dtype=np.int)
        for i in range(voca_num):

            for j in range(sentence_num):
                if labels[j]==0:    # 在正常邮件的数目
                    num_word[0][i] += dataset[j][i]
                elif labels[j]==1:  # 在垃圾邮件中的数目
                    num_word[1][i] += dataset[j][i]
        print('numword:')
        print(num_word)
        self.p_c_0 = np.zeros(voca_num, dtype=np.double)
        self.p_c_1 = np.zeros( voca_num, dtype=np.double)
        for i in range(voca_num):
            self.p_c_0[i] = (num_word[0][i]+1)  / (normal+2)
            self.p_c_1[i] = (num_word[1][i]+1)  / (spam+2)
        print(self.p_c_0)
        print(self.p_c_1)

        pass

    def predict(self, testset):
        """

        :param testset: the dataset to be predicted(still one-hot-encoding)
        :return: an array of labels
        """
        result=[]
        normal_probability=1
        spam_probability=1
        predict_num = len(testset)
        for i in range(predict_num):
            for j in range(self.tag_num):
                if testset[i][j]==1:
                    normal_probability *= self.p_c_0[j]
                    spam_probability *= self.p_c_1[j]
            print(normal_probability,spam_probability)
            if normal_probability>spam_probability:
                result.append(0)
            else:
                result.append(1)
        return result
        pass
        
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1

    return returnVec


if __name__ == '__main__':

    listOPosts, listClasses = loadDataSet()
    VocabList = createVocabList(listOPosts)
    train_dataset = []
    for sentence in listOPosts:
        train_dataset.append(setOfWords2Vec(VocabList, sentence))
    train_dataset = np.array(train_dataset)
    labelset = np.array(listClasses)
    nb_clf = NaiveBayes()
    nb_clf.fit(train_dataset, labelset)
    testset = []
    test1 = setOfWords2Vec(VocabList, ['love', 'my', 'dalmation'])
    test2 = setOfWords2Vec(VocabList, ['stupid', 'garbage'])
    test3 = setOfWords2Vec(VocabList, ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'])
    test4 = setOfWords2Vec(VocabList,['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'])
    test5 = setOfWords2Vec(VocabList,['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'])
    testset.append(test1)
    testset.append(test2)
    testset.append(test3)
    testset.append(test4)
    testset.append(test5)
    testset = np.array(testset)
    result = nb_clf.predict(testset)
    print(result)

