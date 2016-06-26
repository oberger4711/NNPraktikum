#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron
from report.evaluator import Evaluator


def main():
    #one_digit_data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    #print("")
    #runStupidClassifier(one_digit_data)
    #runPerceptronClassifier(one_digit_data)
    #runLogisticClassifier(one_digit_data)
    #one_digit_data = None

    all_digits_data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000, one_hot=False)
    print("")
    runMultilayerClassifier(all_digits_data)

def trainAndEvaluateClassifier(classifier, test_set, verbose=False, graph=False):
    # Train
    print("Train " + classifier.__class__.__name__ + "..")
    classifier.train(verbose=verbose, graph=graph)
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


def runStupidClassifier(data):
    c = StupidRecognizer(data.training_set,
                            data.validation_set,
                            data.test_set)
    trainAndEvaluateClassifier(c, data.test_set)

def runPerceptronClassifier(data):
    c = Perceptron(data.training_set,
                            data.validation_set,
                            data.test_set,
                            learningRate=0.005,
                            epochs=10)
    trainAndEvaluateClassifier(c, data.test_set)

def runLogisticClassifier(data):
    c = LogisticRegression(data.training_set,
                            data.validation_set,
                            data.test_set,
                            learningRate=0.005,
                            epochs=30)
    trainAndEvaluateClassifier(c, data.test_set, verbose=True)

def runMultilayerClassifier(data):
    c = MultilayerPerceptron(data.training_set,
                            data.validation_set,
                            data.test_set,
                            epochs=5,
                            layers=MultilayerPerceptron.createLayers([784, 100, 10], 0.05))
    trainAndEvaluateClassifier(c, data.test_set, verbose=True, graph=True)

if __name__ == '__main__':
    main()
