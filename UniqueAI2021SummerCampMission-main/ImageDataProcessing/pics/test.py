import numpy as np


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



listOPosts, listClasses = loadDataSet()
VocabList = createVocabList(listOPosts)
train_dataset = []
for sentence in listOPosts:
    train_dataset.append(setOfWords2Vec(VocabList, sentence))
train_dataset = np.array(train_dataset)
labelset = np.array(listClasses)
testset = []
test1 = setOfWords2Vec(VocabList, ['love', 'my', 'dalmation'])
test2 = setOfWords2Vec(VocabList, ['stupid', 'garbage'])
testset.append(test1)
testset.append(test2)
testset = np.array(testset)
print(train_dataset)