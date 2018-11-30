"""Neural network classifier with softmax output.

Author: Dylan Blanchard, Sloan Anderson, and Stephen Johnson
Class: CSI-480-01
Assignment: PA 5 -- Supervised Learning
Due Date: Nov 30, 2018 11:59 PM

Certification of Authenticity:
I certify that this is entirely my own work, except where I have given
fully-documented references to the work of others. I understand the definition
and consequences of plagiarism and acknowledge that the assessor of this
assignment may, for the purpose of assessing this assignment:
- Reproduce this assignment and provide a copy to another member of academic
- staff; and/or Communicate a copy of this assignment to a plagiarism checking
- service (which may then retain a copy of this assignment on its database for
- the purpose of future plagiarism checking)

Champlain College CSI-480, Fall 2018
The following code was adapted by Joshua Auerbach (jauerbach@champlain.edu)
from the UC Berkeley Pacman Projects (see license and attribution below).

----------------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""

import util
import numpy as np
import tensorflow as tf
from tensorflow import keras
from classification_method import ClassificationMethod

PRINT = True


class NeuralNetworkClassifier(ClassificationMethod):
    """A neural network classifier with softmax output.

    This Class will perform softmax classification using TensorFlow

    Note that the variable 'datum' in this code refers to a counter of features
    """

    def __init__(self, legal_labels, max_iterations, learning_rates=[0.005]):
        """Initialize the perceptron.

        Args:
            legal_labels: list of legal_labels
            max_iterations: the max number of iterations to train for
            learning_rates: list of learning_rates to try training with
        """
        super().__init__(legal_labels)
        self.type = "ann"
        self.max_iterations = max_iterations
        self.learning_rates = learning_rates
        # note we do not yet have the information to initialize the model,
        # since it depends on the dimensionality of the feature vector
        self.model = None

    def train(self, training_data, training_labels,
              validation_data, validation_labels):
        """Train the neural network.

        The training loop for the neural network classifier passes through the
        training data several times and updates the weight vector for each
        label based on the cross entropy loss.

        You will need to setup the model, compile it, and then fit to the
        training data.

        Make sure to run for self.max_iterations epochs and run in batch mode
        where the batch size is equal to the number of data points.

        self.max_iterations times.

        The model should have very similar architecture to what is shown in
           https://www.tensorflow.org/tutorials/keras/basic_classification

        In general, this example is quite similar to what is shown there,
        except for the source of the data, and the optimizer.

        Important note: this should operate in batch mode,
        using all training_data for each batch, and should use
        keras.optimizers.SGD as in my linear regression demo.
        """
        self.features = list(training_data[0].keys())  # could be useful later
        test_data = np.asarray([np.asarray(list(datum.values()))
                                for datum in validation_data])

        # *** YOUR CODE HERE ***

        for learning_rate in self.learning_rates:
            data_matrix = np.asarray([np.asarray(list(datum.values()))
                                      for datum in training_data])

            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(data_matrix.shape[1],)),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(10, activation=tf.nn.softmax)
            ])

            model.compile(optimizer=keras.optimizers.SGD(
                learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

            model.summary()

            model.fit(data_matrix, training_labels,
                      batch_size=data_matrix.shape[1],
                      epochs=self.max_iterations)

            if self.model is None:
                self.model = model
            else:
                new_loss, new_acc = model.evaluate(test_data,
                                                   validation_labels)
                curr_loss, curr_acc = self.model.evaluate(test_data,
                                                          validation_labels)
                if new_loss < curr_loss and new_acc > curr_acc:
                    self.model = model

    def classify(self, data):
        """Classifies each datum as the label with largest softmax output."""
        # *** YOUR CODE HERE ***
        data_matrix = np.asarray([np.asarray(list(datum.values()))
                                  for datum in data])

        return [np.argmax(pred) for pred in self.model.predict(data_matrix)]

    def find_high_weight_features(self, label, num=100):
        """Return a list of num features with the greatest weight for label."""
        # this function is optional for this classifier, but if you want to
        # visualize the weights of this classifier, you will need to implement
        # it
        util.raise_not_defined()
