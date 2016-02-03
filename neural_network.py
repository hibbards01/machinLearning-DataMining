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

###############################################
# main
#   Main Driver program. This will parse the
#       arguments and make sure they are valid.
#       Then do the implementations based off
#       the arguments.
###############################################
def main(argv):
    # Possible arguments
    inputs = {'-file': None, '-predict': None, '-tree': None, '-help': None, '-nominal': None}
    error = None

    # Loop through the arguments
    for i, input in enumerate(argv):
        # See if this is an input
        if input in inputs:
            # Now is which one it is
            if input == '-tree' or input == '-help' or input == '-nominal':
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
            '\t-nominal, If the data is nominal use this for the script.\n'
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