"""Perceptron implementation for apprenticeship learning in pacman.

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
from perceptron import PerceptronClassifier

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    """A PerceptronClassifier for apprenticeeship learning in pacman."""

    def __init__(self, legal_labels, max_iterations):
        """Initialize the perceptron.

        Args:
            legal_labels: list of legal_labels
            max_iterations: the max number of iterations to train for
        """
        super().__init__(legal_labels, max_iterations)
        self.weights = util.Counter()

    def classify(self, data):
        """Classify the data points.

        Data contains a list of (datum, legal moves)

        Datum is a Counter representing the features of each GameState.
        legal_moves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legal_moves in data:
            vectors = util.Counter()
            for l in legal_moves:
                vectors[l] = self.weights * datum[l]
            guesses.append(vectors.arg_max())
        return guesses

    def train(self, training_data, training_labels, validation_data,
              validation_labels):
        """Train the perceptron."""
        # could be useful later
        self.features = list(training_data[0][0]['Stop'].keys())
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for (datum, legal_moves), label in zip(training_data,
                                                   training_labels):
                # *** YOUR CODE HERE ***
                # Gets the guess action, then updates the weights
                guess = self.classify([(datum, legal_moves)])[0]
                if guess != label:
                    self.weights += datum[label]
                    self.weights -= datum[guess]
