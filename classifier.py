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
    k = 1

    #
    # Member methods
    #
    ###########################################
    # Constructor
    ###########################################
    def __init__(self, trainingData, k):
        self.trainData = trainingData

        if k is not None:
            self.k = int(k)

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

        # Create an array of arrays. This will save the columns
        attributeValues = [[] for i in range(num)]

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
    def saveData(self, dataSet, targets, isFile):
        saveData = [] # Save all the data sets
        size = 0

        # Loop through the data
        for i, data in enumerate (dataSet):
            # Create a new DataSet
            saveData.append(DataSet(data, targets[i]))

            # Save the size
            size += 1

        # Randomize the list
        random.shuffle(saveData)

        # Do we need to change the data?
        if isFile is not None:
            self.changeData(saveData)

        # Grab how many for each set
        train = int(self.train * size)
        test  = train + int(self.train * size) + 1

        # Now save into the trainSet and testSet lists
        self.trainSet = saveData[0:train]
        self.testSet  = saveData[train:test]

        return

    ###########################################
    # changeData
    #   This will change the data to numbers so
    #       so that the data will be easier to use.
    ###########################################
    def changeData(self, saveData):
        # How many attributes are we changing?
        num           = len(saveData[0].data)    # The number of attributes
        attributes    = [{} for i in range(num)] # This will save the attribute values
        attributeNums = [1 for i in range(num)]  # This will increment the number if a new attribute
                                                 # is defined.
        # Loop through the data
        for dataSet in saveData:
            # Loop through the attributes
            for i, attribute in enumerate (dataSet.data):
                # Now check to see if that attribute has been already been seen or not
                if attribute not in attributes[i]:
                    # This is a new attribute
                    attributes[i][attribute] = attributeNums[i]

                    # Increment the value
                    attributeNums[i] += 1

                # Change the value
                dataSet.data[i] = attributes[i][attribute]

        return

    ###########################################
    # runTest
    #   This will train, test, and print out the
    #     results of the algorithm.
    ###########################################
    def runTest(self, k):
        # Create a new instance of the class HardCoded
        hardCoded = HardCoded(self.trainSet, k)

        # Now test it
        results = hardCoded.predict(self.testSet)

        # See how accurate it is
        count = 0
        for i, data in enumerate (self.testSet):
            if data.target == results[i]:
                count += 1

        # Print out the results
        print('\nHere are test results: \n'
            '\tAlgorithm was %0.2f%% accurate\n' % ((count / len(self.testSet) * 100)))

        return

###########################################
# readFile
#   This will read the file that was given
#       by the user.
###########################################
def readFile(fileName, saveData):
    # Open the file
    file = open(fileName, 'r')

    # Read line by line
    for line in file:
        # Make it into an array
        array = line.split(',')

        # Grab the length
        size = len(array)

        # Split the array and save it
        saveData['data'].append(array[0:(size - 1)])
        saveData['targets'].append(array[size - 1].replace('\n', ''))

    # Close the file
    file.close()

    return

###########################################
# Main
#   Main driver of the program
###########################################
def main(argv):
    # Grab the arguments
    inputs = {'-file': None, '-train': None, '-test': None, '-help': False, '-k': None}
    error = None

    # Loop through argv
    for i, input in enumerate (argv):
        if input in inputs:
            if input == '-help':
                inputs[input] = True
            elif input == '-file':
                inputs[input] = argv[i + 1]
            elif (i + 1) < len(argv):
                inputs[input] = float(argv[i + 1])
            else:
                error = '\nError: There was no value provided after the option: %s\nType -help for help\n' % input

    # Make sure they both work
    if inputs['-train'] is not None and inputs['-test'] is not None:
        total = inputs['-train'] + inputs['-test']
        if total < 1.0 or total > 1.0:
            error = '\nError with your train and test values. They must add up to 1.0.\n'

    # See what was passed
    if error is not None:
        print(error)
    elif inputs['-help']:
        print('\nCommand line arguments for classifier.py:\n\n'
            '    py classifier.py [options] [value]\n\n'
            '    Options:\n'
            '\t-train, Give a percent of what this should be trained on. OPTIONAL. Default 0.7\n'
            '\t-test,  Give a percent of what this should be tested on. OPTIONAL. Default 0.3\n'
            '\t-k,     Give a number for how many k-neighbors should be used to predict the value. DEFAULT 1\n'
            '\t-file,  Give a .csv file for the data that you want to test against. OPTIONAL.\n'
            '\t        DEFAULT Iris data will be tested.\n'
            '\t-help,  Show this.\n')
    else:
        file = {'data': [], 'targets': []}

        # Import the data set
        if inputs['-file'] is not None:
            readFile(inputs['-file'], file)
        else:
            # Grab from the iris
            from sklearn import datasets
            iris = datasets.load_iris();

            # Save it
            file['data'] = iris.data
            file['targets'] = iris.target

        # Now create the classifier class
        classifier = Classifier(inputs['-train'], inputs['-test'])

        # Now give the data to the classifier
        classifier.saveData(file['data'], file['targets'], inputs['-file'])

        # Finally test the algorithm
        classifier.runTest(inputs['-k'])

    return

# Invoke the program
if __name__ == "__main__":
    main(sys.argv)