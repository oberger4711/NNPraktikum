
import sys
import logging
import numpy as np

from sklearn.metrics import accuracy_score

# from util.activation_functions import Activation
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 cost='crossentropy', learning_rate=0.01, epochs=50):

        """
        A digit-7 recognizer based on logistic regression algorithm

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

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_task = output_task  # Either classification or regression
        self.output_activation = output_activation
        self.cost = cost

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers
        self.input_weights = input_weights

        # Build up the network from specific layers
        # Here is an example of a MLP acting like the Logistic Regression
        self.layers = []
        output_activation = "sigmoid"
        #self.layers.append(LogisticLayer(self.training_set.input.shape[1], 4, learning_rate=learning_rate))
        self.layers.append(LogisticLayer(4, 10, learning_rate=learning_rate))

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self.get_layer(0)

    def _get_output_layer(self):
        return self.get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        """

        # Here you have to propagate forward through the layers
        # and remember the activation values of each layer

        # Hidden layers
        inp_next_layer = inp
        for layer in self.layers:
            inp_next_layer = layer.forward(inp_next_layer)

        return inp_next_layer

    def _compute_error(self, expected_outp):
        """
        Compute the total error of the network and propagate back through the net

        Returns
        -------
        ndarray :
            a numpy array (1, nOut) containing the output of the layer
        """

        next_derivatives, next_weights = self.layers[-1].computeOutDerivative(expected_outp)
        for hidden_layer in reversed(self.layers[:-1]):
            next_derivatives, next_weights = hidden_layer.computeDerivative(next_derivatives, next_weights)

    def _update_weights(self):
        """
        Update the weights of the layers
        """

        for layer in self.layers:
            layer.updateWeights()

    def train(self, verbose=False, graph=False):
        """
        Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        if len(self.layers) == 0:
            logging.e("Trying to train MLP without any layer.")
        for epoch in range(0, self.epochs):
            self._train_one_epoch()
            if verbose:
                accuracy = accuracy_score(self.validation_set.label, self.evaluate(self.validation_set))
                logging.info("New validation accuracy after epoch %i: %.1f%%", epoch + 1, accuracy * 100)

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for expected_label, inp in zip(self.training_set.label, self.training_set.input):
            # Encode expected label into output shape.
            expected_outp = np.zeros(10)
            expected_outp[expected_label] = 1

            self._feed_forward(inp)
            self._compute_error(expected_outp)
            self._update_weights()

    def classify(self, test_instance):
        return self._feed_forward(test_instance).argmax()

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
            test = self.test_set.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))
