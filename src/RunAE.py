#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven

from model.denoising_ae import DenoisingAutoEncoder
from model.mlp import MultilayerPerceptron
from model.logistic_layer import LogisticLayer

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    # data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
    #                  one_hot=True, target_digit='7')

    # NOTE:
    # Comment out the MNISTSeven instantiation above and
    # uncomment the following to work with full MNIST task
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                      one_hot=False)

    # NOTE:
    # Other 1-digit classifiers do not make sense now for comparison purpose
    # So you should comment them out, let alone the MLP training and evaluation

    # Train the classifiers #
    n_encoder_neurons = 50
    myDAE = DenoisingAutoEncoder(data.training_set,
                                 data.validation_set,
                                 data.test_set,
                                 n_hidden_neurons=n_encoder_neurons,
                                 learning_rate=0.05,
                                 epochs=10)

    print("Train autoencoder..")
    myDAE.train(verbose=True)
    print("Done..")

    # Multi-layer Perceptron
    # NOTES:
    # Now take the trained weights (layer) from the Autoencoder
    # Feed it to be a hidden layer of the MLP, continue training (fine-tuning)
    # And do the classification

    # Correct the code here
    layers = []
    # Add auto envoder hidden layer.
    layers.append(LogisticLayer(data.training_set.input.shape[1], n_encoder_neurons, weights=myDAE.get_weights(), cost="mse", activation="sigmoid", learning_rate=0.05))
    # Add another hidden layer just like in the previous exercise.
    n_second_hidden_neurons = 100
    layers.append(LogisticLayer(n_encoder_neurons, n_second_hidden_neurons, cost="mse", activation="sigmoid", learning_rate=0.05))
    # Add output classifier layer with one neuron per digit.
    n_out_neurons = 10
    layers.append(LogisticLayer(n_second_hidden_neurons, n_out_neurons, cost="crossentropy", activation="softmax", learning_rate=0.05))
    myMLPClassifier = MultilayerPerceptron(data.training_set,
                                           data.validation_set,
                                           data.test_set,
                                           layers=layers,
                                           epochs=10)

    print("Train MLP..")
    myMLPClassifier.train(verbose=True)
    print("Done..")
    print("")

    # Evaluate
    print("Evaluate..")
    mlpPred = myMLPClassifier.evaluate()
    print("Done..")
    print("")

    print("Results:")
    evaluator = Evaluator()

    print("")

    # print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    # evaluator.printAccuracy(data.test_set, stupidPred)

    # print("\nResult of the Perceptron recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.test_set, perceptronPred)

    # print("\nResult of the Logistic Regression recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.test_set, lrPred)

    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.test_set, mlpPred)

    # Draw
    plot = PerformancePlot("DAE + MLP on MNIST task")
    plot.draw_performance_epoch(myMLPClassifier.performances,
                                myMLPClassifier.epochs)

if __name__ == '__main__':
    main()
