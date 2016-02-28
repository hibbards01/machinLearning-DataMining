#!/usr/bin/env Rscript
########################################################
# Program:
#   Support Vector Machine
# Author:
#   Samuel Hibbard
# Summary:
#   This will implement the support vector machine algorithm.
########################################################

# Grab the arguments that are passed
args <- commandArgs(trailingOnly = TRUE)

# Set some values
inputs <- c('', '')
names(inputs) <- c('-file', '-class')

# Grab the file if it is greater then 0
if (length(args) > 1) {
  inputs['-file'] = args[2]
}

# Include the LIBSVM package
library (e1071)


# For your assignment, you'll need to read from a CSV file.
# Conveniently, there is a read.csv() function that can be used like so:
data = NULL
if (inputs['-file'] != "") {
  data = read.csv(inputs['-file'], head=TRUE, sep=",")
} else {
  # Load our old friend, the Iris data set
  # Note that it is included in the default datasets library
  library(datasets)
  data(iris)
  data = iris
}

# Partition the data into training and test sets
# by getting a random 30% of the rows as the testRows
allRows = 1:nrow(data)
testRows = sample(allRows, trunc(length(allRows) * 0.3))

# The test set contains all the test rows
data_test = data[testRows,]

# The training set contains all the other rows
data_train = data[-testRows,]

# Train an SVM model
# Tell it the attribute to predict vs the attributes to use in the prediction,
#  the training data to use, and the kernal to use, along with its hyperparameters.
#  Please note that "Species~." contains a tilde character, rather than a minus
# model = svm(Species~., data = data_train, kernel="radial", gamma = 0.001, cost = 10, type="C")
model = svm(Rings~., data = data_train, kernel="radial", gamma = 0.001, cost = 10, type="C")
# model = svm(Class~., data = data_train, kernel="radial", gamma = 0.001, cost = 10, type="C")

# Use the model to make a prediction on the test set
# Notice, we are not including the last column here (our target)
# prediction = predict(model, data_test[,-5])
prediction = predict(model, data_test[,-9])
# prediction = predict(model, data_test[,-14])

# Produce a confusion matrix
# confusion_matrix = table(pred = prediction, true = data_test$Species)
confusion_matrix = table(pred = prediction, true = data_test$Rings)
# confusion_matrix = table(pred = prediction, true = data_test$Class)

# Calculate the accuracy, by checking the cases that the targets agreed
# agreement = prediction == data_test$Species
agreement = prediction == data_test$Rings
# agreement = prediction == data_test$Class
accuracy = prop.table(table(agreement))

# Print our results to the screen
print(confusion_matrix)
print(accuracy)