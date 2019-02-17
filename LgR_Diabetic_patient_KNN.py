import numpy as np
from operator import itemgetter
import csv, os
import pandas

class KNNClassifier(object):
    def __init__(self):
        self.training_features = None # number of kick and kisses
        self.training_labels = None # movies type
        # testdata placeholder
        self.test_features = None # kick, kisses
        #build meaningful result
        self.elegantResult = "Most likely. {0} kick and '{1}' kisses is of type/class '{2}'."


    def loadTrainingDataFromFile(self, file_path):
        if file_path is not None and os.path.exists(file_path):
            features = [[],[],[],[],[],[],[],[],[]]
            self.training_labels = []
            name = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
            reader = pandas.read_csv(file_path, names=name)
            print(reader)
            preg=reader['preg'].values
            plas = reader['plas'].values
            pres = reader['pres'].values
            skin = reader['skin'].values
            test = reader['test'].values
            mass = reader['mass'].values
            pedi = reader['pedi'].values
            age = reader['age'].values
            self.training_labels=reader['class'].values
            self.training_features = np.array(list(zip(preg,plas,pres,skin,test,mass,pedi,age)))
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


def IspredictDiabeties():
    My_test_data=[9,80,62,0,0,22.5,0.142,36]
    instance=KNNClassifier()
    instance.loadTrainingDataFromFile('indians-diabetes.data.csv')
    classOfTestData=instance.classifyTestData(test_data=My_test_data,k=5)
    #classOfTestData=instance.classifyTestData(test_data=None,k=3)
    print("Predict Diabetic patient Type=",classOfTestData)



if __name__=='__main__':
   IspredictDiabeties()
