########################################################
# Program:
#   ID3 Decision Tree Algorithm
# Author:
#   Samuel Hibbard
# Summary:
#   This will take some training data and use the algorithm
#     to determine the test data.
########################################################

# Import some files
import sys
import numpy as np
import random

##########################################
# DecisionTree
##########################################
class DecisionTree:
    #
    # Member variables
    #
    root = {}

    ###########################################
    # Constructor
    ###########################################
    def __init__(self, data, targetCol):
        # Set the available attriubtes
        availableAtts = [i for i in range(len(data[0]) - 1)]

        # Begin the training
        self.root = self.train(self.root, data, availableAtts, targetCol)

    ###########################################
    # calcEntropy
    #   This will find the best entropy among the
    #       given attributes.
    ###########################################
    def calcEntropy(self, data, availableAtts, targetCol):
        # Grab all the targets
        targets = []
        totalSize = 0
        for row in data:
            targets.append(row[targetCol])
            totalSize += 1

        # Loop through each of the attributes
        gain = 0.0
        for attribute in availableAtts:
            values = {}

            # Now loop through the data to grab it all
            for i, row in enumerate (data):
                if row[attribute] not in values:
                    values.update({row[attribute] : []})

                values[row[attribute]].append(targets[i])

            # Now loop though the values
            for value in values:
                # Collect how many times that they appear
                from collections import Counter
                collect = Counter(values[value])

                # Grab the size
                size = sum(collect.values())

                # Loop through the collect
                entropySum = 0
                for key in collect:
                    # Now add up the entropies
                    entropySum += -(collect[key] / size) * np.log2(collect[key] / size)

                # Finally add up the gain
                gain += (size / totalSize) * entropySum

            print(gain)
            print("DONE")
        return

    ###########################################
    # train
    #   Create the tree based off the training data.
    ###########################################
    def train(self, node, data, availableAtts, targetCol):
        self.calcEntropy(data, availableAtts, targetCol)
        return node

    ###########################################
    # predict
    #   Predict the values based off the testing data.
    ###########################################
    def predict(self, test):
        return

##########################################
# Classifier
##########################################
class Classifier:
    #
    # Member variables
    #
    data       = []
    predictCol = -1
    inputs     = []

    ###########################################
    # Constructor
    ###########################################
    def __init__(self, inputs):
        # Read the data
        self.readFile(inputs['-file'])

        if inputs['-predict'] is None:
            self.predictCol = 4
        else:
            self.predictCol = inputs['-predict']

        # Now change it
        self.changeData()

        # Save the inputs
        self.inputs = inputs

    ###########################################
    # readFile
    #   This will read the file that was given
    #       by the user. Or read the iris data.
    ###########################################
    def readFile(self, fileName):
        if fileName is not None:
            # Open the file
            file = open(fileName, 'r')

            # Read line by line
            for line in file:
                # Make it into an array
                array = line.split(',')

                # Grab the length
                size = len(array)

                # Split the array and save it
                self.data.append(array)

            # Close the file
            file.close()
        else:
            # Else use the iris data
            from sklearn import datasets
            iris = datasets.load_iris()

            # Finally save the iris data
            values = np.array(iris.data)
            targets = iris.target

            # Add the targets
            for i, array in enumerate(values):
                self.data.append(np.append(values[i], targets[i]))

        return

    ###########################################
    # changeData
    #   This will change the data so that it will
    #       work for the decision tree.
    ###########################################
    def changeData(self):
        # Grab the number of attributes
        num = len(self.data[0])

        # Create an array of arrays. This will save the columns
        attributeValues = [[] for i in range(num)]

        # Split the attributes by it's columns
        for attributes in self.data:
            # Now loop through the columns
            for i, attribute in enumerate (attributes):
                # Now save it
                attributeValues[i].append(attribute)

        # Now loop through the attributes
        newValues = []
        for i, attribute in enumerate (attributeValues):
            # Don't change the predictions
            if i != self.predictCol:
                # Find low, max, and middle
                low = min(attribute)
                m = max(attribute)
                middle = (low + m) / 2

                # Grab the med, high values
                med  = (low + middle) / 2
                high = (middle + m) / 2

                # Save the values
                newValues.append([med, high])
            else:
                newValues.append([])

        # Now change the original data
        for r, row in enumerate (self.data):
            for c, col in enumerate (row):
                if newValues[c]:
                    newValue = 1

                    # Check the values
                    if col < newValues[c][0]:
                        newValue = -1
                    elif col < newValues[c][1]:
                        newValue = 0

                    # Assign the new value
                    row[c] = newValue

        return

    ###########################################
    # startProgram
    #   This will take the arguments and run
    #       the program according to the inputs.
    ###########################################
    def startProgram(self):
        # Randomize the data
        random.shuffle(self.data)

        # Grab the size
        size = len(self.data)

        # Grab how many for each set
        trainIndex = int(0.7 * size)
        testIndex  = int(size + 1)

        # Now save into the trainSet and testSet lists
        train = self.data[0:trainIndex]
        test  = self.data[trainIndex:testIndex]

        # Finally start the test
        tree = DecisionTree(train, self.predictCol)

        # Now predict the values
        tree.predict(test)

        return

###############################################
# main
#   Main Driver program. This will parse the
#       arguments and make sure they are valid.
#       Then do the implementations based off
#       the arguments.
###############################################
def main(argv):
    # Possible arguments
    inputs = {'-file': None, '-predict': None, '-tree': None, '-help': None}
    error = None

    # Loop through the arguments
    for i, input in enumerate(argv):
        # See if this is an input
        if input in inputs:
            # Now is which one it is
            if input == '-tree' or input == '-help':
                inputs[input] = True
            elif (i + 1) < len(argv):
                if input == '-file':
                    inputs[input] = argv[i + 1]
                elif input == '-predict':
                    inputs[input] = int(argv[i + 1])
            else:
                error = '\nError: No value given for argument %s.\nType -help for help.\n\n' % input

    # If file is defined make sure that the predict is defined.
    if inputs['-file'] is not None and error is None:
        if inputs['-predict'] is None:
            error = '\nError: option -predict must be given if using a file.\nType -help for help.\n\n'

    # Now do the operation
    if error is not None:
        print(error)
    elif inputs['-help']:
        print('\nCommand line arguments for decision_tree.py:\n\n'
            '    py decision_tree.py [options] [value]\n\n'
            '    Options:\n'
            '\t-file,    Give a .csv file for the data that you want to test against. OPTIONAL.\n'
            '\t          If given then option -predict must be given.\n'
            '\t          DEFAULT Iris data will be tested.\n'
            '\t-predict, The column of the data that you want to predict. OPTIONAL.\n'
            '\t-tree,    Test the data with an existing implementation.\n'
            '\t-help,    Show this.\n')
    else:
        # Run the program
        classifier = Classifier(inputs)
        classifier.startProgram()

    return

# This will start the program in main
if __name__ == '__main__':
    main(sys.argv)