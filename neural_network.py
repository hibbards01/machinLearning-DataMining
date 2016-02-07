#!/usr/bin/python
########################################################
# Program:
#   Neural Network
# Author:
#   Samuel Hibbard
# Summary:
#   This program will build a neural network classifier.
#     the neural network will learn based off the training
#     data and will then test against the test data to see
#     how accurate it is.
########################################################

# Import some fles
import sys
import numpy as np
from copy import deepcopy
import random

###########################################
# normalize
#   This will change the data based off the
#       standard deviation and mean.
###########################################
def normalize(dataSet):
    # Grab the number of attributes
    num = len(dataSet[0])

    # Create an array of arrays. This will save the columns
    attributeValues = [[] for i in range(num)]

    # Split the attributes by it's columns
    for attributes in dataSet:
        # Now loop through the columns
        for i, attribute in enumerate (attributes):
            # Now save it
            attributeValues[i].append(attribute)

    # Grab the standard deviation and mean
    stndDev = np.std(attributeValues, axis=1)
    mean = np.mean(attributeValues, axis=1)

    # Now save the zscore
    for attributes in dataSet:
        for i, attribute in enumerate (attributes):
            attributes[i] = (attribute - mean[i]) / stndDev[i]

    return dataSet

###########################################
# readFile
#   This will read the file that was given
#       by the user.
###########################################
def readFile(fileName, data, targets):
    # Open the file
    file = open(fileName, 'r')

    # Read line by line
    for line in file:
        # Make it into an array
        array = line.split(',')

        # Grab the length
        size = len(array)

        # Split the array and save it
        data.append(array[0:(size - 1)])
        targets.append(array[size - 1].replace('\n', ''))

    # Close the file
    file.close()

    return

###########################################
# randomize
#   This will randomize the data and targets.
###########################################
def randomize(data, targets):
    # Add the targets to the data
    combine = list(zip(data, targets))

    # Now randomize it
    random.shuffle(combine)

    # Now split them again
    data, targets = zip(*combine)

    return

##########################################
# NeuralNetwork
#   This will be the main classifier.
##########################################
class NeuralNetwork:
    #
    # Member variables
    #
    nodes = []

    #
    # Member methods
    #
    ###########################################
    # Constructor
    #   This will build the nodes and put all the
    #       weights into the nodes variable.
    ###########################################
    def __init__(self, nodes, weights):


###############################################
# main
#   Main Driver program. This will parse the
#       arguments and make sure they are valid.
#       Then do the implementations based off
#       the arguments.
###############################################
def main(argv):
    # Possible arguments
    inputs = {'-file': None, '-nodes': None, '-help': None, '-net': None}
    error = None

    # Loop through the arguments
    for i, input in enumerate(argv):
        # See if this is an input
        if input in inputs:
            # Now is which one it is
            if input == '-net' or input == '-help':
                inputs[input] = True
            elif (i + 1) < len(argv):
                if input == '-file':
                    inputs[input] = argv[i + 1]
                elif input == '-nodes':
                    inputs[input] = int(argv[i + 1])
            else:
                error = '\nError: No value given for argument %s.\nType -help for help.\n\n' % input

    # Make sure nodes was created
    if inputs['-nodes'] is None:
        error = '\nError: You must use the -nodes option.\n'

    # Now do the operation
    if error is not None:
        print(error)
    elif inputs['-help']:
        print('\nCommand line arguments for decision_tree.py:\n\n'
            '    py decision_tree.py [options] [value]\n\n'
            '    Options:\n'
            '\t-file,    Give a .csv file for the data that you want to test against. OPTIONAL.\n'
            '\t          DEFAULT Iris data will be tested.\n'
            '\t-nodes,   Specify how many nodes in the network. REQUIRED\n'
            '\t-net,     Test data with exisiting implementation.\n'
            '\t-help,    Show this.\n')
    else:
        # Grab the data
        data = []
        targets = []

        if inputs['-file'] is not None:
            readFile(inputs['-file'], data, targets)
        else:
            # Load the iris data
            from sklearn import datasets
            iris = datasets.load_iris();

            # Now save them
            data = iris.data
            targets = iris.target

        # Now hurry and normalize the data
        data = normalize(data)

        # Randomize it
        randomize(data, targets)

        # Now start the neural network
        network = NeuralNetwork(inputs['-nodes'], len(data[0]))

    return

# This will start the program in main
if __name__ == '__main__':
    main(sys.argv)