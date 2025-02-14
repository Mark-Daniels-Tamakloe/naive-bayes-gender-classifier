#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naïve Bayes Classifier Implementation
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY

def naivebayesCL(x, y):
    """
    Constructs a Naïve Bayes classifier in a linear form.

    Parameters:
    x : ndarray
        Feature matrix with d dimensions and n samples (dxn).
    y : ndarray
        Label vector (-1 or +1) of size (1xn).

    Returns:
    w : ndarray
        Weight vector used for classification.
    b : float
        Bias term.
    """

    # Convert input to NumPy matrix form
    X = np.matrix(x)
    d, n = X.shape  # Extract feature and sample dimensions

    # Compute P(Y) for both class labels
    prob_y_pos, prob_y_neg = naivebayesPY(x, y)
    
    # Compute P(X | Y) for all features
    prob_x_given_y_pos, prob_x_given_y_neg = naivebayesPXY(x, y)

    # Flatten probabilities for easier computation
    prob_x_given_y_pos = np.asarray(prob_x_given_y_pos).flatten()
    prob_x_given_y_neg = np.asarray(prob_x_given_y_neg).flatten()

    # Compute weight vector: log-odds ratio between probabilities
    w = np.log(prob_x_given_y_pos / (1 - prob_x_given_y_pos)) - np.log(prob_x_given_y_neg / (1 - prob_x_given_y_neg))

    # Compute bias: class log-ratio plus feature probabilities adjustment
    b = np.log(prob_y_pos / prob_y_neg) + np.sum(np.log(1 - prob_x_given_y_pos)) - np.sum(np.log(1 - prob_x_given_y_neg))

    # Ensure w is a column vector
    w = w.reshape(-1, 1)

    return w, b
