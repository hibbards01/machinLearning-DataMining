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
import json
import math

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

        # Convert strings to nums
        array = [float(i) for i in array]

        # Split the array and save it
        data.append(array[0:(size - 1)])
        targets.append(array[size - 1])

    # Close the file
    file.close()

    return

##########################################
# NeuralNetwork
#   This will be the main classifier.
##########################################
class NeuralNetwork:
    #
    # Member variables
    #
    network = [] # This will hold the weights
    inputs  = [] # This will hold the number of inputs for each layer

    #
    # Member methods
    #
    ###########################################
    # Constructor
    #   This will build the nodes and put all the
    #       weights into the nodes variable.
    ###########################################
    def __init__(self, network, weights, classes):
        # Grab how many layers and nodes
        numNodes = [int(index) for index in network.split(',')]
        layers   = len(numNodes) + 1 # For the final layer

        # First create the layers
        self.network = [[] for i in range(layers)]

        # Now add the nodes to each layer
        for l, layer in enumerate(self.network):
            # Grab how many nodes
            num = numNodes[l] if l < layers - 1 else classes

            # Now create the nodes
            for n in range(num):
                layer.append([])

        # Add one to the weights for a bias node
        weights += 1
        numNodes = [i + 1 for i in numNodes]

        # Put all the inputs together
        self.inputs = [weights] + numNodes

        # Finally add the weights
        for l, layer in enumerate(self.network):
            for node in layer:
                for i in range(self.inputs[l]):
                    node.append([random.uniform(-1,1)])

    ###########################################
    # activationFunction
    #   This will do the calculation of wether
    #       or not a node activated or not.
    ###########################################
    def activationFunction(self, outputs):
        # Go through each of the outputs
        for value in outputs[0]:
            # Now do the equation
            value[0] = 1 / (1 + math.exp(value[0]))

        return outputs

    ###########################################
    # updateWeights
    #   Update the weights if the feed forward
    #
    ###########################################
    def updateWeights(self, inputs, outputs):
        return

    ###########################################
    # feedNetwork
    #   This will train the neural network based
    #       of the data
    ###########################################
    def feedNetwork(self, data, targets, train=True):
        # Add the bias node for all the inputs
        newData = []
        for i, row in enumerate(data):
            newData.append(np.append(row, -1))

        # First loop through the data
        predictions = []
        for i, row in enumerate(newData):
            # Grab the inputs and reshape it
            inputs = np.array(row).reshape(1, self.inputs[0])

            # Start looping through the layers
            # saveInputs  = []
            # saveOutputs = []
            for l, layer in enumerate(self.network):
                # Skip the first layer since the inputs were from the data set
                if l > 0:
                    # See if the nodes were activiated
                    outputs = self.activationFunction(inputs)

                    # Add the bias to the new inputs
                    newInputs = np.append(outputs, -1)

                    # Then reshape it
                    inputs = np.array(newInputs).reshape(1, self.inputs[l])

                # Save the inputs
                # saveInputs.append(inputs)

                # Do the dot product
                inputs = np.dot(inputs, layer)

            # Now find out which class had won
            outputs = list(inputs[0])
            winner = outputs.index(max(outputs))

            # Save the predictions if we are testing
            if train == False:
                predictions.append(winner)

        return predictions

    ###########################################
    # train
    #   This will train the neural network.
    ###########################################
    def train(self, train, targets):
        # Start the training
        predictions = self.feedNetwork(train, targets)

        return

    ###########################################
    # test
    #   Test out the neural network. This will
    #       print how accurate it is.
    ###########################################
    def test(self, test, targets):
        # Start the testing
        predictions = self.feedNetwork(test, targets, False)

        return predictions

##########################################
# Classifier
#   This will hold the neural network.
#       It will get the data ready and
#       train and test the Neural Network.
##########################################
class Classifier:
    #
    # Member variables
    #
    data    = []
    targets = []
    neuralNetwork = None

    #
    # Member methods
    #
    ###########################################
    # Constructor
    ###########################################
    def __init__(self, data, targets):
        # Normalize the data
        data = self.normalize(data)

        # Randomize the data
        self.randomize(data, targets)

    ###########################################
    # randomize
    #   This will randomize the data and targets.
    ###########################################
    def randomize(self, data, targets):
        # Add the targets to the data
        combine = list(zip(data, targets))

        # Now randomize it
        random.shuffle(combine)

        # Now split them again
        for row in combine:
            dataRow, target = row

            self.data.append(dataRow)
            self.targets.append(target)

        return

    ###########################################
    # normalize
    #   This will change the data based off the
    #       standard deviation and mean.
    ###########################################
    def normalize(self, dataSet):
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
    # run
    #   Run the program. This will train the Neural
    #       Network then test it.
    ###########################################
    def run(self, network):
        # Split the data
        train = int(len(self.data) * 0.7)
        test  = len(self.data)

        trainSet     = self.data[0:train]
        testSet      = self.data[train:test]
        trainTargets = self.targets[0:train]
        testTargets  = self.targets[train:test]

        # Grab how many classes
        classes = len(set(self.targets))

        # Now start the neural network
        self.network = NeuralNetwork(network, len(self.data[0]), classes)

        # Train it
        self.network.train(trainSet, trainTargets)   # Training here first

        # Now test it
        predictions = self.network.test(testSet, testTargets)

        # See how accurate it is
        count = 0
        for i, target in enumerate (testTargets):
            if  target == predictions[i]:
                count += 1

        # Print out the results
        print('\nHere are test results: \n'
            '\tNeuralNetwork algorithm was %0.2f%% accurate\n' % ((count / len(testTargets) * 100)))

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
    inputs = {'-file': None, '-exist': None, '-help': None, '-net': None}
    error = None

    # Loop through the arguments
    for i, input in enumerate(argv):
        # See if this is an input
        if input in inputs:
            # Now is which one it is
            if input == '-exist' or input == '-help':
                inputs[input] = True
            elif (i + 1) < len(argv):
                if input == '-file':
                    inputs[input] = argv[i + 1]
                elif input == '-net':
                    inputs[input] = argv[i + 1]
            else:
                error = '\nError: No value given for argument %s.\nType -help for help.\n\n' % input

    # Make sure nodes was created
    if inputs['-net'] is None and inputs['-help'] is None:
        error = '\nError: You must use the -net option.\nType -help for help.\n\n'

    # Now do the operation
    if error is not None:
        print(error)
    elif inputs['-help']:
        print('\nCommand line arguments for decision_tree.py:\n\n'
            '    py decision_tree.py [options] [value]\n\n'
            '    Options:\n'
            '\t-file,    Give a .csv file for the data that you want to test against. OPTIONAL.\n'
            '\t          DEFAULT Iris data will be tested.\n'
            '\t-net,     Specify how many hidden layers and nodes in the network. REQUIRED\n'
            '\t          EXAMPLE: -net 2,3,3 This will create 3 layers with 2 nodes in the first,\n'
            '\t          3 in the second, and 3 in the last layer. No spaces between commas.\n'
            '\t-exist,   Test data with exisiting implementation.\n'
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

        # Create the classifier
        classifier = Classifier(data, targets)

        # Now run the classifier
        classifier.run(inputs['-net'])
    return

# This will start the program in main
if __name__ == '__main__':
    main(sys.argv)