import numpy as np

from util.activation_functions import Activation


class LogisticLayer():
    """
    A layer of neural

    Parameters
    ----------
    n_in: int: number of units from the previous layer (or input data)
    n_out: int: number of units of the current layer (or output)
    cost: string: cost function to be used
    activation: string: activation function of every units in the layer
    is_classifier_layer: bool:  to do classification or regression

    Attributes
    ----------
    n_in : positive int:
        number of units from the previous layer
    n_out : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activation_string : string
        the name of the activation function
    is_classifier_layer: bool
        to do classification or regression
    deltas : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    learning_rate : float
        Learning rate to use in weight updates
    """

    def __init__(self, n_in, n_out, weights=None, cost="mse",
                 activation='sigmoid', is_classifier_layer=False, learning_rate=0.1):

        # Get activation function from string
        self.activation_string = activation
        self.activation = Activation.get_activation(self.activation_string)
        self.activation_derivative = Activation.get_derivative(self.activation_string)

        self.cost = cost
        # TODO: Make this more modular.
        assert self.cost == "mse" or self.cost == "crossentropy"

        self.n_in = n_in
        self.n_out = n_out

        self.inp = np.ndarray((n_in + 1))
        self.inp[0] = 1
        self.outp = np.ndarray((n_out, 1))
        self.deltas = np.zeros((n_out, 1))

        # You can have better initialization here
        if weights is None:
            self.weights = np.random.rand(n_in + 1, n_out) / 10
        else:
            assert weights.shape == (n_in + 1, n_out)
            self.weights = weights
        self.threshold = np.random.rand() / 10

        self.learning_rate = learning_rate

        self.is_classifier_layer = is_classifier_layer

        # Some handy properties of the layers
        self.size = self.n_out
        self.shape = self.weights.shape

    def UnloadFunctions(self):
        # Workaround for pickle not being able to persist functions.
        self.activation = None
        self.activation_derivative = None

    def LoadFunctions(self):
        # Workaround for pickle not being able to persist functions.
        self.activation = Activation.get_activation(self.activation_string)
        self.activation_derivative = Activation.get_derivative(self.activation_string)

    def forward(self, inp):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        inp : ndarray
            a numpy array (1,n_in) containing the input of the layer

        Change outp
        -------
        outp: ndarray
            a numpy array (1,n_out) containing the output of the layer
        """

        self.inp = np.ndarray(inp.shape[0] + 1)
        self.inp[0] = 1
        self.inp[1:] = inp
        self.outp = self._fire(inp)

        return self.outp

    def computeDerivative(self, next_derivatives, next_weights):
        """
        Compute the derivatives (backward)

        Parameters
        ----------
        next_derivatives: ndarray
            a numpy array containing the derivatives from next layer
        next_weights : ndarray
            a numpy array containing the weights from next layer

        Change deltas
        -------
        deltas: ndarray
            a numpy array containing the partial derivatives on this layer
        """

        for neuron in range(0, self.n_out):
            self.deltas = self.activation_derivative(self.outp) * np.dot(next_derivatives, next_weights[neuron, :])

        return self.deltas, self.weights[1:, :]

    def computeOutDerivative(self, expected_outp):
        """
        Compute the derivatives if this is an output layer

        Parameters
        ----------
        expected_outp: ndarray
            a numpy array containing the expected result for the given input

        Change deltas
        -------
        deltas: ndarray
            a numpy array containing the partial derivatives on this layer
        """

        if self.cost == "crossentropy":
            self.deltas = expected_outp - self.outp
        elif self.cost == "mse":
            self.computeDerivative(expected_outp - self.outp, np.ones(self.n_out))
        else:
            logging.e("Unknown cost function '%s'.", self.cost)

        return self.deltas, self.weights[1:, :]

    def updateWeights(self):
        """
        Update the weights of the layer
        """

        for neuron in range(0, self.n_out):
            self.weights[:, neuron] += self.learning_rate * self.deltas[neuron] * self.inp

    def _fire(self, inp):
        return self.activation(np.dot(self.inp, self.weights))

    def getOutput(self):
        return self.outp
