########################################################
# Program:
#   apriori.r
# Author:
#   Samuel Hibbard
# Summary:
#   Working with the Association Rule Mining.
########################################################

# Grab the arguments that are passed
args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 3) {
    sup = as.numeric(args[1])
    conf = as.numeric(args[2])
    len = as.numeric(args[3])

    # Grab the library
    library(arules)

    # Grab the data now
    data(Groceries)
    data = Groceries

    # Now inspect the transactions
    rules <- apriori(data = data,
                    parameter = list(support = sup,
                                    confidence = conf,
                                    minlen = len))

    # Now show the rules
    print("__________RULES__________")
    inspect(sort(rules, by = "support", decreasing = FALSE))

    # Find rules for a particular product
    # newRules <- subset(rules, items %in% "ice cream")
    # inspect(newRules)
} else {
    print("ERROR: you must pass in these arguments as numbers: <support> <confidence> <minlen>")
}