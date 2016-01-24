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
        saveData.append(array)

    # Close the file
    file.close()

    return

###########################################
# startProgram
#   This will take the arguments and run
#       the program according to the inputs.
###########################################
def startProgram(inputs):
    # What data are we using?
    data = []
    if inputs['-file'] is not None:
        readFile(inputs['-file'], data)
    else:
        # Else use the iris data
        from sklearn import datasets
        iris = datasets.load_iris()

        # Finally save the iris data
        values = np.array(iris.data)
        targets = iris.target

        # Add the targets
        for i, array in enumerate(values):
            data.append(np.append(values[i], targets[i]))

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
            '\t-file,    Give a .csv file for the data that you want to test against.'
            '\t          If given then option -predict must be given. OPTIONAL.\n'
            '\t-predict, The column of the data that you want to predict. OPTIONAL.\n'
            '\t          DEFAULT Iris data will be tested.\n'
            '\t-tree,    Test the data with an existing implementation.\n'
            '\t-help,    Show this.\n')
    else:
        # Run the program
        startProgram(inputs)

    return

# This will start the program in main
if __name__ == '__main__':
    main(sys.argv)