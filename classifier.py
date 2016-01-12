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

import sys
import random

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
    targetName = ''

    #
    # Member methods
    #
    ###########################################
    # Constructor
    ###########################################
    def __init__(self, data, target, targetName):
        self.data       = data
        self.target     = target
        self.targetName = targetName

##########################################
# HardCoded
##########################################
class HardCoded:
    #
    # Member variables
    #


    #
    # Member methods
    #
    ###########################################
    # Constructor
    ###########################################
    def __init__(self, trainingData):
        self.training(trainingData)

    ###########################################
    # training
    #   This will use the data in order to train
    #       itself.
    ###########################################
    def training(self, data):
        # Training...
        return

    ###########################################
    # predict
    #   This will now test the algorithm. Based
    #       upon what it was trained.
    ###########################################
    def predict(self, data):
        results = []

        # For loop through the data
        for i in data:
            results.append(0)

        return results


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
    trainSet = [] # This will hold all the trainSet values
    testSet  = [] # This will hold all the testSet values
    train    = 0  # What percentage of the list do you want to train?
    test     = 0  # What percentage of the list do you want to test?

    #
    # Member methods
    #
    ###########################################
    # Constructor
    ###########################################
    def __init__(self, train=0.7, test=0.3):
        self.train    = train
        self.test     = test

    ###########################################
    # saveData
    #   This will read and save the data it was
    #       be given. It will also randomize it
    #       and save it to the trainSet and testSet
    #       variables.
    ###########################################
    def saveData(self, irisData):
        iris       = [] # Save all the data sets
        targetName = 0  # Index for the target names
        size       = 0

        # Loop through the data
        for i, data in enumerate (irisData.data):
            # Now see where we are at
            if i % 50 == 0 and i > 0:
                targetName += 1 # Increment the number if we are there

            # Create a new DataSet
            iris.append(DataSet(data, irisData.target[i], irisData.target_names[targetName]))

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
        hardCoded = HardCoded(self.trainSet)

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
    # Import the data set
    from sklearn import datasets
    irisData = datasets.load_iris();

    # Now create the classifier class
    classifier = Classifier()

    # Now give the data to the classifier
    classifier.saveData(irisData)

    # Finally test the algorithm
    classifier.runTest()

# Invoke the program
if __name__ == "__main__":
    main(sys.argv)