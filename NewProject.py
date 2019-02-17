
import numpy as np
from operator import itemgetter
import csv, os
import pandas

class KNNClassifier(object):
    def __init__(self):
        self.training_features = None
        self.training_labels = None
        # testdata placeholder
        self.test_features = None
        #build meaningful result



    def loadTrainingDataFromFile(self):
            features = [[],[],[],[]]
            self.training_labels = []
            url='https://goo.gl/QnHW4g'
            name = ['sepal-lenght', 'sepal-width', 'petal-lenght', 'petal-width', 'class']
            data = pandas.read_csv(url, names=name)
            print(data.shape)
            print(data)
            sepal_l=data['sepal-lenght'].values
            sepal_w = data['sepal-width'].values
            petal_l = data['petal-lenght'].values
            petal_w = data['petal-width'].values
            self.training_labels=data['class'].values
            self.training_features = np.array(list(zip(sepal_l,sepal_w,petal_l,petal_w)))
            print(self.training_features)
            print(self.training_labels)

    def classifyTestData(self, test_data = None, k = 0):
        print ("classifyTestData: test_data :", test_data)
        if test_data is not None:
            self.test_features = np.array(test_data, dtype=float)
        print ("classify test data: self.test_features :", self.test_features)

        #ensure we have training data, training labels, test_data and number of 'k'
        if self.test_features is not None and self.training_features is not None and self.training_features is not None and k > 0:
            print ("classify test data: self.test_features :", self.test_features)
            print ("training_features:\n", self.training_features)
            print ("training_labels:\n", self.training_labels)
            featureVectorSize = self.training_features.shape[0]
            print ("featureVectorSize = ",featureVectorSize)
            tileOfTestData = np.tile(self.test_features, (featureVectorSize,1))
            print ("after tile\n", tileOfTestData)
            diffMat = self.training_features - tileOfTestData
            sqDiffMat = diffMat**2
            sqDistances = sqDiffMat.sum(axis=1)
            distances = sqDistances**0.5
            print (distances)
            sortedDistanceIndices = distances.argsort()
            print (sortedDistanceIndices)
            print (self.training_labels)
            classCount = {}
            for i in range(k):
                print (i,sortedDistanceIndices)
                voteILabel = self.training_labels[sortedDistanceIndices[i]]
                print ("voteILabel: ", voteILabel)
                classCount[voteILabel] = classCount.get(voteILabel,0) + 1
            print ("classcount",classCount)
            sortedClassCount = sorted(classCount.items(),key=itemgetter(1), reverse=True)
            return sortedClassCount[0][0]
        else:
            return "Can't determine result for empty test-data"


def IspredictType():
    My_test_data=[4.1,2.9,1.4,0.1]
    instance=KNNClassifier()
    instance.loadTrainingDataFromFile()
    classOfTestData=instance.classifyTestData(test_data=My_test_data,k=5)
    #classOfTestData=instance.classifyTestData(test_data=None,k=3)
    print("Predict flower Type=",classOfTestData)



if __name__=='__main__':
   IspredictType()
