#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=10)
    # Uncomment this to run Logistic Neuron Layer
    myLRClassifier = LogisticRegression(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)

    # Train the classifiers
    print("=========================")
    print("Training..")
    print("")

    print("Train Stupid Classifier..")
    myStupidClassifier.train()
    print("Done..")
    print("")

    print("Train Perceptron..")
    # Change parameter to True for verbose output.
    myPerceptronClassifier.train(True)
    print("Done..")
    print("")

    print("Train Logistic Regression..")
    myLRClassifier.train()
    print("Done..")
    print("")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    perceptronPred = myPerceptronClassifier.evaluate()
    lrPred = myLRClassifier.evaluate()

    # Report the result
    print("=========================")
    print("Results..")
    print("")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)
    print("")

    print("Result of the Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, perceptronPred)
    print("")

    print("Result of the Logistic Regression recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    #evaluator.printAccuracy(data.testSet, lrPred)
    print("")


if __name__ == '__main__':
    main()
