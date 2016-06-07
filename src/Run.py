#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator


def main():
    runStupidClassifier()
    runPerceptronClassifier()
    runLogisticClassifier()

def trainAndEvaluateClassifier(classifier, test_set, verbose=False, graph=False):
    # Train
    print("Train " + classifier.__class__.__name__ + "..")
    classifier.train()
    print("Done..")
    print("")

    # Evaluate
    print("Evaluate..")
    pred = classifier.evaluate()
    print("Done..")
    print("")

    # Results
    evaluator = Evaluator()
    print("Result:")
    # evaluator.printComparison(data.test_set, stupidPred)
    evaluator.printAccuracy(test_set, pred)
    print("")


def runStupidClassifier():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    stupidClassifier = StupidRecognizer(data.training_set,
                                          data.validation_set,
                                          data.test_set)
    trainAndEvaluateClassifier(stupidClassifier, data.test_set)

def runPerceptronClassifier():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    perceptronClassifier = Perceptron(data.training_set,
                                        data.validation_set,
                                        data.test_set,
                                        learningRate=0.005,
                                        epochs=10)
    trainAndEvaluateClassifier(perceptronClassifier, data.test_set)

def runLogisticClassifier():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    logisticClassifier = LogisticRegression(data.training_set,
                                        data.validation_set,
                                        data.test_set,
                                        learningRate=0.005,
                                        epochs=30)
    trainAndEvaluateClassifier(logisticClassifier, data.test_set)

if __name__ == '__main__':
    main()
