# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
from sklearn.metrics import accuracy_score

from util.loss_functions import *
from model.logistic_layer import LogisticLayer
from model.auto_encoder import AutoEncoder

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class DenoisingAutoEncoder(AutoEncoder):
    """
    A denoising autoencoder.
    """

    def __init__(self, training_set, validation_set, test_set, n_hidden_neurons=100, noise_ratio=0.3, learning_rate=0.05, error=DifferentError(), epochs=30):
        """
         Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """
        self.noise_ratio = noise_ratio
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.error = error

        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set

        self.layers = []
        n_input_neurons = training_set.input.shape[1]
        self.layers.append(LogisticLayer(n_input_neurons, n_hidden_neurons, cost="mse", activation="sigmoid", learning_rate=self.learning_rate))
        self.layers.append(LogisticLayer(n_hidden_neurons, n_input_neurons, cost="mse", activation="sigmoid", learning_rate=self.learning_rate))

    def train(self, verbose=True):
        """
        Train the denoising autoencoder
        """
        for epoch in range(0, self.epochs):
            self._train_one_epoch()

            if verbose:
                error = self.evaluate(self.validation_set)
                logging.info("New denoising error after epoch %i: %.4f", epoch + 1, error)


    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for inp in self.training_set.input:
            inp_with_noise = self._add_noise(inp)

            self._feed_forward(inp_with_noise)
            self._compute_error(inp)
            self._update_weights()

    def _feed_forward(self, inp):
        inp_next_layer = inp
        for layer in self.layers:
            inp_next_layer = layer.forward(inp_next_layer)

        return inp_next_layer

    def _compute_error(self, target):
        next_derivatives, next_weights = self.layers[-1].computeOutDerivative(target)
        for hidden_layer in reversed(self.layers[:-1]):
            next_derivatives, next_weights = hidden_layer.computeDerivative(next_derivatives, next_weights.T)

    def _update_weights(self):
        for layer in self.layers:
            layer.updateWeights()

    def get_weights(self):
        """
        Get the weights (after training)
        """
        return self.layers[0].weights

    def evaluate(self, test=None):
        if test is None:
            test = self.test_set.input

        error = 0
        for test_instance in test:
            test_instance_with_noise = self._add_noise(test_instance)
            outp = self.reconstruct(test_instance_with_noise)
            #error += self.error.calculate_error(test_instance, outp)
            error += abs(np.sum(test_instance - outp))

        return error

    def reconstruct(self, test_instance):
        return self._feed_forward(test_instance)

    def _add_noise(self, instance):
        total_size = instance.shape[0]
        zeros_size = int(total_size * self.noise_ratio)
        mask = np.ones(total_size - zeros_size)
        mask = np.insert(mask, 0, np.zeros(zeros_size))
        np.random.shuffle(mask)

        return instance * mask

