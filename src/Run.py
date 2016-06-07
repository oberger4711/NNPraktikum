#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator


def main():
    #runStupidClassifier()
    #runPerceptronClassifier()
    runLogisticClassifier()


def runStupidClassifier():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    myStupidClassifier = StupidRecognizer(data.training_set,
                                          data.validation_set,
                                          data.test_set)
    # Train
    print("Train Stupid Classifier..")
    myStupidClassifier.train()
    print("Done..")
    print("")

    # Evaluate
    print("Evaluate..")
    stupidPred = myStupidClassifier.evaluate()
    print("Done..")
    print("")

    # Results
    evaluator = Evaluator()
    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.test_set, stupidPred)
    evaluator.printAccuracy(data.test_set, stupidPred)
    print("")

def runPerceptronClassifier():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    myPerceptronClassifier = Perceptron(data.training_set,
                                        data.validation_set,
                                        data.test_set,
                                        learningRate=0.005,
                                        epochs=10)

    # Train
    print("Train Perceptron Classifier..")
    myPerceptronClassifier.train(True)
    print("Done..")
    print("")

    # Evaluate
    print("Evaluate..")
    perceptronPred = myPerceptronClassifier.evaluate()
    print("Done..")
    print("")

    # Results
    evaluator = Evaluator()
    print("Result of the Perceptron recognizer:")
    # evaluator.printComparison(data.test_set, perceptronPred)
    evaluator.printAccuracy(data.test_set, perceptronPred)
    print("")

def runLogisticClassifier():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    myLRClassifier = LogisticRegression(data.training_set,
                                        data.validation_set,
                                        data.test_set,
                                        learningRate=0.005,
                                        epochs=30)

    # Train
    print("Train Logistic Classifier..")
    myLRClassifier.train(graph=True)
    print("Done..")
    print("")
    
    # Evaluate
    print("Evaluate..")
    lrPred = myLRClassifier.evaluate()
    print("Done..")
    print("")

    # Results
    evaluator = Evaluator()
    print("Result of the Logistic Regression recognizer:")
    # evaluator.printComparison(data.test_set, perceptronPred)
    evaluator.printAccuracy(data.test_set, lrPred)
    print("")

if __name__ == '__main__':
    main()
