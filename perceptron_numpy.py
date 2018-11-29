"""Optimized perceptron implementation using numpy.

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


import numpy as np
import util
from classification_method import ClassificationMethod
PRINT = True


class OptimizedPerceptronClassifier(ClassificationMethod):
    """Optimized Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    """

    def __init__(self, legal_labels, max_iterations):
        """Initialize the perceptron.

        Args:
            legal_labels: list of legal_labels
            max_iterations: the max number of iterations to train for
        """
        super().__init__(legal_labels)
        self.type = "perceptron_numpy"
        self.max_iterations = max_iterations
        # note we do not yet have the information to initialize the weight
        # matrix, since it depends on the dimensionality of the feature vector
        self.weights = None
        self.features = None

    def train(self, training_data, training_labels, validation_data,
              validation_labels):
        """Train the perceptron.

        The training loop for the perceptron passes through the training data
        several times and updates the weight vector for each label based on
        classification errors.

        See the assignment description for details.

        The data will still come in with each data point as a counter from
        features to values for those features (and thus represents a vector
        of values), so this code first converts
        the data to a numpy array.

        In this case, we will not use validation_data or validation_labels

        Important: Do not change this function!
        """
        # now we can initialize the weights
        if self.weights is None:
            # features list could be useful later
            self.features = list(training_data[0].keys())

            self.weights = np.zeros((len(self.features),
                                     len(self.legal_labels)))

        data_matrix = np.asarray([np.asarray(list(datum.values()))
                                  for datum in training_data])

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for datum, label in zip(data_matrix, training_labels):
                self.perform_update(datum, label)

    def perform_update(self, datum, label):
        """Update the weights based on a single data point.

        Args:
            datum: input data point (now a numpy array)
            label: label of datum

        You must use the provided self.weights numpy array.
        """
        # *** YOUR CODE HERE ***
        results = np.dot(datum, self.weights)
        winner = np.argmax(results)
        if winner == label:
            pass
        else:
            self.weights[:,winner] = self.weights[:,winner] - datum
            self.weights[:,label] = self.weights[:,label] + datum

    def classify(self, data):
        """Classify each datum.

        Each datum is classified as the label that most closely matches
        the prototype vector for that label.

        See the project description for details.

        Recall that a datum is a util.counter...
        so the data list must first be converted to a numpy array
        (see example in train above), then you must return the prediction
        for each entry.
        """
        if self.weights is None:
            raise Exception("the perceptron must be trained "
                            "before data can be classified")

        # *** YOUR CODE HERE ***
        guesses = []
        data_matrix = np.asarray([np.asarray(list(datum.values()))
                                  for datum in data])
        for datum in data_matrix:
            results = np.dot(datum, self.weights)
            winner = np.argmax(results)
            guesses.append(winner)
        return guesses

    def find_high_weight_features(self, label, num=100):
        """Return a list of num features with the greatest weight for label.

        Args:
            label: label to find features with greatest weight for
            num: number of features to return (default 100)

        Hint 1: self.features stores the list of features names.  Here you will
        have to find which rows contain the largest values in the column of
        self.weights corresponding to the given label, and then return the
        feature names for those rows

        Hint 2: to get the keys of a dictionary sorted by their value you can
        do

            sorted([key for key in dictionary.keys()],
                   key=lambda k: dictionary[k])

        You can also set some other function or lambda expression as the sort
        key
        """
        # *** YOUR CODE HERE ***
        weight_list = enumerate(self.weights[:,label].tolist())
        end_list = sorted(weight_list, key=lambda x: x[1])
        feat_list = []
        for i in end_list[-num:]:
            feat_list.append(self.features[i[0]])
        return feat_list
