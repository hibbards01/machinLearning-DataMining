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
from sknn.mlp import Classifier, Layer
import matplotlib.pyplot as plt

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
    # Member methods
    #
    ###########################################
    # Constructor
    #   This will build the nodes and put all the
    #       weights into the nodes variable.
    ###########################################
    def __init__(self, network, weights, classes, n):
        # Grab how many layers and nodes
        numNodes = [int(index) for index in network.split(',')]
        layers   = len(numNodes) + 1 # For the final layer

        # Save the size
        self.size = layers

        # Set the learning rate
        self.n = n

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
            value[0] = 1 / (1 + math.exp(-value[0]))

        return outputs

    ###########################################
    # outputError
    #   This will calculate the output error for
    #       all the output errors
    ###########################################
    def outputErrors(self, output, target):
        # Loop through each node
        errors = []
        for n, node in enumerate(output[0]):
            # Grab the target value
            value = 1 if n == target else 0

            # Now do this math: aj(1 - aj)(aj - value)
            error = node[0] * (1 - node[0]) * (node[0] - value)

            # Save the error
            errors.append(error)

        return errors

    ###########################################
    # hiddenError
    #   Same as outputError but for the hidden
    #       layers instead.
    ###########################################
    def hiddenErrors(self, weights, outputs, errors):
        # Loop through the current node's outputs
        newErrors = []

        # Loop through all the outputs except for the last one
        size = len(outputs[0])
        for n in range(size - 1):
            # Sum up the weights times the errors on the right layer
            # wjk * dk
            # Loop through the weights
            error = 0
            for w, weight in enumerate(weights):
                error += weight[n][0] * errors[w]

            # Compute final answer: aj(1 - aj)error
            finalError = outputs[0][n] * (1 - outputs[0][n]) * error
            newErrors.append(finalError)

        return newErrors

    ###########################################
    # updateWeights
    #   Update the weights if the feed forward
    #       produces not the desired output.
    ###########################################
    def updateWeights(self, inAndOut, target):
        # Grab how many layers in the network
        size = len(self.network)

        # Loop through the layers reversed
        errors = None  # dj
        weights = None # This will save the previous weights that haven't been updated yet
        for l, layer in reversed(list(enumerate(self.network))):
            # Is this the outer layer?
            if (l + 1) == size:
                # Then do this special case
                errors = self.outputErrors(inAndOut[-1], target)
            else:
                # Otherwise do this case for the hidden layers
                errors = self.hiddenErrors(weights, inAndOut[l + 1], errors)

            # Save the old weights
            weights = layer

            # Loop through the nodes
            # Do this equation: wij = wij - n * dj * ai
            for n, node in enumerate(layer):
                # Compute this n * dj
                rhs = self.n * errors[n]

                # Loop through each weight
                for w, weight in enumerate(node):
                    # Times this to ai
                    rhs *= inAndOut[l][0][w]

                    # Finally change the weight!
                    node[w][0] = weight[0] - rhs
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

            inAndOut = [] # Save the input/outputs after each feed
            inAndOut.append(inputs)
            # Start looping through the layers
            for l in range(self.size):
                # Skip the first layer since the inputs were from the data set
                if l > 0:
                    # See if the nodes were activiated
                    outputs = self.activationFunction(inputs)

                    # Add the bias to the new inputs
                    newInputs = np.append(outputs, -1)

                    # Then reshape it
                    inputs = np.array(newInputs).reshape(1, self.inputs[l])

                    # Save the new inputs
                    inAndOut.append(inputs)

                # Do the dot product
                inputs = np.dot(inputs, self.network[l])

            # Save the last outputs
            inputs = self.activationFunction(inputs)
            inAndOut.append(inputs)

            # Now find out which class had won
            outputs = list(inputs[0])
            winner = outputs.index(max(outputs))

            # Save the predictions
            predictions.append(winner)

            # Only train the network if we are training
            if train:
                self.updateWeights(inAndOut, targets[i])

        return predictions

    ###########################################
    # train
    #   This will train the neural network.
    ###########################################
    def train(self, train, targets):
        # Start the training
        predictions = self.feedNetwork(train, targets)

        return predictions

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
    # Member methods
    #
    ###########################################
    # Constructor
    ###########################################
    def __init__(self, data, targets, epochs, n):
        # Create member variables
        self.train = []
        self.test  = []
        self.trainTargets = []
        self.testTargets  = []
        self.epochs = epochs if epochs is not None else 100
        self.n = n if n is not None else 0.2

        # Normalize the data
        data = self.normalize(data)

        # Split the data
        train = int(len(data) * 0.7)
        test  = len(data)

        self.train        = data[0:train]
        self.test         = data[train:test]
        self.trainTargets = targets[0:train]
        self.testTargets  = targets[train:test]

        # Randomize the data
        self.randomize(self.train, self.trainTargets, True)
        self.randomize(self.test, self.testTargets, False)

    ###########################################
    # randomize
    #   This will randomize the data and targets.
    ###########################################
    def randomize(self, data, targets, trainData):
        # Add the targets to the data
        combine = list(zip(data, targets))

        # Now randomize it
        random.shuffle(combine)

        # Now split them again
        newData = []
        newTargets = []
        for row in combine:
            dataRow, target = row
            newData.append(dataRow)
            newTargets.append(target)

        if trainData:
            self.train = newData
            self.trainTargets = newTargets
        else:
            self.test = newData
            self.testTargets = newTargets
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
    # training
    #   Train the neural network
    ###########################################
    def training(self):
        # Start the epochs
        save = []
        for e in range(self.epochs):
            # Grab the predictions
            predictions = self.network.train(self.train, self.trainTargets)

            # How accurate was it?
            a = self.printAccuracy(predictions, self.trainTargets)
            save.append(a)

            # Randomize the data again
            self.randomize(self.train, self.trainTargets, True)

        plt.plot(save)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

        return

    ###########################################
    # printAccuracy
    #   This will show how accurate it was.
    ###########################################
    def printAccuracy(self, predictions, targets):
        save = 0
        count = 0
        for i, target in enumerate (targets):
            if target == predictions[i]:
                count += 1

        a = (count / len(targets) * 100)
        save = a
        print('%0.2f%%' % (a))

        return save

    ###########################################
    # run
    #   Run the program. This will train the Neural
    #       Network then test it.
    ###########################################
    def run(self, network, exist):
        predictions = []
        if exist:
            pass
            # nn = Classifier(layers=[Layer("Rectifier", units=100), Layer("Softmax")], learning_rate=0.02, n_iter=10)
            # nn.fit(self.train, self.trainTargets)
            # predictions = clf.predict(self.test)
        else:
            # Grab how many classes
            classes = len(set(self.trainTargets))

            # Now start the neural network
            self.network = NeuralNetwork(network, len(self.train[0]), classes, self.n)

            # Train it
            self.training()

            # Now test it
            predictions = self.network.test(self.test, self.testTargets)

        # Print out the results
        print('\nHere are test results: \n'
            'NeuralNetwork algorithm was: ')

        # See how accurate it is
        save = self.printAccuracy(predictions, self.testTargets)

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
    inputs = {'-file': None, '-exist': None, '-help': None, '-net': None, '-e': None, '-l': None}
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
                elif input == '-e':
                    inputs[input] = int(argv[i + 1])
                elif input == '-l':
                    inputs[input] = int(argv[i + 1])
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
            '\t-l,       How fast should it learn? OPTIONAL. DEFAULT 0.2\n'
            '\t-e,       How many epochs for training the network?. DEFAULT 100\n'
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
        classifier = Classifier(data, targets, inputs['-e'], inputs['-l'])

        # Now run the classifier
        classifier.run(inputs['-net'], inputs['-exist'])
    return

# This will start the program in main
if __name__ == '__main__':
    main(sys.argv)