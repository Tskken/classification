"""This file contains the abstract class ClassificationMethod.

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


class ClassificationMethod:
    """Abstract super class.

    As such, you need not add any code to this file.  You can write
    all of your implementation code in the files for the individual
    classification methods.
    """

    def __init__(self, legal_labels):
        """Initialize with given legal_labels.

        For digits dataset, the set of legal labels will be 0,1,..,9
        For faces dataset, the set of legal labels will be
            0 (non-face) or 1 (face)
        """
        self.legal_labels = legal_labels

    def train(self, training_data, training_labels,
              validation_data, validation_labels):
        """Train the classifier.

        Two sets of labeled data are passed in: a large training set
        and a small validation set.

        Many types of classifiers have a common training structure in practice:
        using training data for the main supervised training loop but tuning
        certain parameters with a small held-out validation set.

        For some classifiers (naive Bayes, MIRA), you will need to return the
        parameters' values after training and tuning step.

        To make the classifier generic to multiple problems, the data should
        be represented as lists of Counters containing feature descriptions and
        their counts.
        """
        util.raise_not_implemented_error()

    def classify(self, data):
        """Return a list of labels, each drawn from the set of legal labels.

        To make the classifier generic to multiple problems, the data should be
        represented as lists of Counters containing feature descriptions and
        their counts.
        """
        util.raise_not_implemented_error()
