# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

__author__ = "Ole Berger"  # Adjust this when you copy the file
__email__ = "unexz@student.kit.edu"  # Adjust this when you copy the file

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, learningRate=0.1, epochs=50):

        self.learningRate = learningRate
        self.epochs = min(epochs, len(train.input))

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and 0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1] + 1)/10

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
	# Here you have to implement the Perceptron Learning Algorithm
        # to change the weights of the Perceptron
        for i in range(0, self.epochs):
            label, data = self.trainingSet.label[i], self.trainingSet.input[i]
            error = label - self.classify(data)
            weightStep = self.learningRate * error
            self.weight[1:] += weightStep * np.array(data)
            self.weight[0] += weightStep
            if verbose:
                logging.info("New validation accuracy after adjusting weights: %f", self.validateAccuracy())
    
    def validateAccuracy(self):
        correctClassifications = 0
        for label, data in zip(self.validationSet.label, self.validationSet.input):
            if self.classify(data) == label:
                correctClassifications += 1
        return float(correctClassifications) / len(self.validationSet.input)

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input

        # Here is the map function of python - a functional programming concept
        # It applies the "classify" method to every element of "test"
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        # I already implemented it for you to see how you can work with numpy
        return Activation.sign(np.dot(np.array(input), self.weight[1:]) + self.weight[0])
