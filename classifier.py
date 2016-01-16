#!/usr/bin/python
########################################################
# Program:
#   classifier.py
# Author:
#   Samuel Hibbard
# Summary:
#   This will be the start of the Classifier. This will
#     take in a data set and give an ouptut of what it
#     evaluated from the data set.
#######################################################

# Import some libraries
import sys
import random
import csv
import math
import operator

##########################################
# DataSet
#   This will hold the data, target, and
#       target name for the one type of set.
##########################################
class DataSet:
    #
    # Member variables
    #
    data       = []
    target     = ''

    #
    # Member methods
    #
    ###########################################
    # Constructor
    ###########################################
    def __init__(self, data, target):
        self.data   = data
        self.target = target

##########################################
# HardCoded
##########################################
class HardCoded:
    #
    # Member variables
    #
    trainData = []
    k = 0

    #
    # Member methods
    #
    ###########################################
    # Constructor
    ###########################################
    def __init__(self, trainingData, k):
        self.trainData = trainingData
        self.k = k

        # Normalize the data
        self.trainData = self.normalize(self.trainData)

    ###########################################
    # normalize
    #   This will change the data based off the
    #       standard deviation and mean.
    ###########################################
    def normalize(self, dataSet):
        # Import the library for the calucations
        import numpy as np

        # Grab the number of attributes
        num = len(dataSet[0].data)

        # Create an dictionary array. This will save the columns
        attributeValues = [[]] * num

        # Split the attributes by it's columns
        for attributes in dataSet:
            # Now loop through the columns
            for i, attribute in enumerate (attributes.data):
                # Now save it
                attributeValues[i].append(attribute)

        # Grab the standard deviation and mean
        stndDev = np.std(attributeValues, axis=1)
        mean = np.mean(attributeValues, axis=1)

        # Now save the zscore
        for attributes in dataSet:
            for i, attribute in enumerate (attributes.data):
                attributes.data[i] = (attribute - mean[i]) / stndDev[i]

        return dataSet

    ###########################################
    # findDistance
    ###########################################
    def findDistance(self, instance1, instance2):
        distance = 0

        # Now loop through all the instance1
        for i, instance in enumerate (instance1):
            # Grab the distance
            distance += pow((instance - instance2[i]), 2)

        # Return the distance
        return math.sqrt(distance)

    ###########################################
    # findNeighbors
    #   This will find the closest neighbors to
    #       the test set.
    ###########################################
    def findNeighbors(self, testInstance):
        distances = []

        # Loop through all the training set
        for instance in self.trainData:
            # First grab the distance
            distance = self.findDistance(testInstance, instance.data)

            # Now save the distance
            distances.append((instance.target, distance))

        # Now sort the distances by the value
        distances.sort(key=operator.itemgetter(1))

        # Grab the neighbors closest to the testInstance
        neighbors = []
        for n in range(self.k):
            neighbors.append(distances[n][0])

        return neighbors

    ###########################################
    # getVote
    #   This will vote on who the test data is.
    ###########################################
    def getVote(self, neighbors):
        votes = {}

        # Loop through the neighbors
        for response in neighbors:
            # Save response
            if response in votes:
                votes[response] += 1
            else:
                votes[response] = 1

        # Sort the votes
        sortedVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)

        # Grab the winner!
        return sortedVotes[0][0]

    ###########################################
    # predict
    #   This will now test the algorithm. Based
    #       upon what it was trained.
    ###########################################
    def predict(self, testData):
        # Make some variables
        predictions = []

        # Normalize the testData
        testData = self.normalize(testData)

        # # Loop throught the test data and grab the predictions
        for testInstance in testData:
            # First find the neighbors
            neighbors = self.findNeighbors(testInstance.data)

            # Now vote on what that data set is
            vote = self.getVote(neighbors)

            # Save the prediction
            predictions.append(vote)

        return predictions


##########################################
# Classifier
#   This will be the main driver class of
#       this program. It will hold the test
#       and train data. This will also call
#       the class 'HardCoded' which will be
#       the one that gets trained. It will
#       then determine how accurate the class
#       was.
##########################################
class Classifier:
    #
    # Member variables
    #
    trainSet = []  # This will hold all the trainSet values
    testSet  = []  # This will hold all the testSet values
    train    = 0.7 # What percentage of the list do you want to train?
    test     = 0.3 # What percentage of the list do you want to test?

    #
    # Member methods
    #
    ###########################################
    # Constructor
    ###########################################
    def __init__(self, train, test):
        if train is not None:
            self.train = train

        if test is not None:
            self.test  = test

        # Now fix one or the other
        if train is not None and test is None:
            self.test = 1.0 - train
        elif test is not None and train is None:
            self.train = 1.0 - test

    ###########################################
    # saveData
    #   This will read and save the data it was
    #       be given. It will also randomize it
    #       and save it to the trainSet and testSet
    #       variables.
    ###########################################
    def saveData(self, irisData):
        iris       = [] # Save all the data sets
        size       = 0

        # Loop through the data
        for i, data in enumerate (irisData.data):
            # Create a new DataSet
            iris.append(DataSet(data, irisData.target[i]))

            # Save the size
            size += 1

        # Randomize the list
        random.shuffle(iris)

        # Grab how many for each set
        train = int(self.train * size)
        test  = train + int(self.train * size) + 1

        # Now save into the trainSet and testSet lists
        self.trainSet = iris[0:train]
        self.testSet  = iris[train:test]

        return

    ###########################################
    # runTest
    #   This will train, test, and print out the
    #     results of the algorithm.
    ###########################################
    def runTest(self):
        # Create a new instance of the class HardCoded
        hardCoded = HardCoded(self.trainSet, 3)

        # Now test it
        results = hardCoded.predict(self.testSet)

        # See how accurate it is
        count = 0
        for i, data in enumerate (self.testSet):
            if data.target == results[i]:
                count += 1

        # Print out the results
        print('\nHere are test results: \n'
            '\tAlgorithm was %f accurate\n' % (count / len(self.testSet)))

        return

###########################################
# Main
#   Main driver of the program
###########################################
def main(argv):
    # Grab the arguments
    helpI = False
    train = None
    test  = None
    csv  = None
    for i, input in enumerate (argv):
        # See what was inputed
        if input == '-train' and (i + 1) < len(argv):
            train = float(argv[i + 1])
        elif input == '-test' and (i + 1) < len(argv):
            test = float(argv[i + 1])
        elif input == '-help':
            helpI = True
        elif input == '-file' and (i + 1) < len(argv):
            csv = argv[i + 1]


    # Make sure they both work
    if train is not None and test is not None:
        if (train + test) < 1.0 or (train + test) > 1.0:
            print('\nError with your train and test values. They must add up to 1.0.\n\n')
            return

    # See what was passed
    if helpI:
        print('\nCommand line arguments for classifier.py:\n\n'
            '    py classifier.py [options] [value]\n\n'
            '    Options:\n'
            '\t-train, Give a percent of what this should be trained on. OPTIONAL. Default 0.7\n'
            '\t-test,  Give a percent of what this should be tested on. OPTIONAL. Default 0.3\n'
            '\t-file,  Give a .csv file for the data that you want to test against. OPTIONAL.\n'
            '\t        DEFAULT Iris data will be tested.\n'
            '\t-help,  Show this.\n\n')
    else :
        # Import the data set
        data = []
        if csv is not None:
            data = []
        else :
            from sklearn import datasets
            data = datasets.load_iris();

        # Now create the classifier class
        classifier = Classifier(train, test)

        # Now give the data to the classifier
        classifier.saveData(data)

        # Finally test the algorithm
        classifier.runTest()

    return

# Invoke the program
if __name__ == "__main__":
    main(sys.argv)