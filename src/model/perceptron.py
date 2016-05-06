# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

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
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and 0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/10

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
	# Here you have to implement the Perceptron Learning Algorithm
        # to change the weights of the Perceptron
        for label, data in zip(self.trainingSet.label, self.trainingSet.input):
            if self.classify(data) != label:
                self.weight = np.asarray(np.array(self.weight) + self.learningRate * np.array(data))
                if verbose:
                    print "Adjusted weights after wrong classification."
            elif verbose:
                    print "Correct classification."

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
        # Hard limiter.
        return self.fire(testInstance) > 0

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
        return Activation.sign(np.dot(np.array(input), self.weight))
