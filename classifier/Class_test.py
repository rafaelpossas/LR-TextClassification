import csv
import random
import numpy as np
import math

class Base:

    def load_data(self,filename):
        lines = csv.reader(open(filename, "rt"))
        dataset = []
        for row in lines:
            del row[0]
            row = [float(x) for x in row]
            dataset.append(row)
        return dataset

    def load_target(self,targetname):
        lines = csv.reader(open(targetname, "rt"))
        dataset = []
        for row in lines:
            del row[0]
            row = [x for x in row]
            dataset.append(row)
        return dataset

    def separateByClass(self,dataset,targetset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            classValue = targetset[i][0]
            if (classValue not in separated):
                separated[classValue] = []
            separated[classValue].append(vector)
        return separated

    def train_test_split(self,dataset,targetset, splitRatio):
        trainSize = int(len(dataset) * splitRatio)
        X_test = []
        y_test = []
        X = list(dataset)
        y = list(targetset)
        while len(X) > trainSize:
            index = random.randrange(len(X))
            X_test.append(X.pop(index))
            y_test.append(y.pop(index))

        return [X, X_test,y,y_test]


    def create_summaries(self, dataset):
        summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*dataset)]
        return summaries

    def summarize_class(self,X_train,y_train):
        separated = self.separateByClass(X_train,y_train)
        summaries = {}
        for classValue,instances in separated.items():
            summaries[classValue] = self.create_summaries(instances)
        return summaries

class NaiveBayes(Base):

    def __init__(self,dataFileName,targetFileName,split_ratio=0.67):
        self.data_fileName = dataFileName
        self.target_fileName = targetFileName
        self.dataset = []
        self.targetset = []
        self.X_train,self.X_test,self.y_train,self.y_test = [],[],[],[]
        self.split_ratio = split_ratio
        self.summaries = []
        self.training_set = []
        self.test_set = []

    def fit(self,dataset=None,targetset=None):
        if (dataset is None or targetset is None) :
            self.dataset = self.load_data(self.data_fileName)
            self.targetset = self.load_target(self.target_fileName)
        else:
            self.dataset = dataset
            self.targetset = targetset

        self.X_train, self.X_test,self.y_train,self.y_test = self.train_test_split(self.dataset,self.targetset, self.split_ratio)
        self.summaries = self.summarize_class(self.X_train,self.y_train)

    def calculateProbability(self,x, mean, stdev):
        try:
            exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
            probability = (1 / ((math.sqrt(2 * math.pi) * stdev)) * exponent)
        except:
            return 0
        return probability


    def calculateClassProbabilities(self,summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
        return probabilities

    def get_prediction(self,dataset=None):
        if(dataset is None):
            dataset = self.dataset
        probabilities = self.calculateClassProbabilities(self.summaries, dataset)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def predict(self,testSet=None):
        predictions = []
        if(testSet is None):
            testSet = self.X_test
        for i in range(len(testSet)):
            result = self.get_prediction(testSet[i])
            predictions.append(result)
        return predictions

    def getAccuracy(self, predictions):
        correct = 0
        for x in range(len(self.y_test)):
            if self.y_test[x][0] == predictions[x]:
                correct += 1
        return (correct / float(len(self.y_test))) * 100.0






def main():
    filename = '../assignment1_2016S1/training_data.csv'
    splitRatio = 0.67
    nb = NaiveBayes('../assignment1_2016S1/training_data.csv','../assignment1_2016S1/training_labels.csv')
    nb.fit()
    pred = nb.predict()
    print(nb.getAccuracy(pred))
    # dataset = loadCsv(filename)
    # trainingSet, testSet = splitDataset(dataset, splitRatio)
    # print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # # prepare model
    # summaries = summarizeByClass(trainingSet)
    # # test model
    # predictions = getPredictions(summaries, testSet)
    # accuracy = getAccuracy(testSet, predictions)
    # print('Accuracy: {0}%'.format(accuracy))

main()