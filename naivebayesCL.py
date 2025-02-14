#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
@author: MN (categorical/Bernoulli NB)
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY

def naivebayesCL(x, y):
    """
    Implementation of a Bernoulli Naïve Bayes classifier.
    
    :param x: Feature matrix (dxn), binary (0 or 1)
    :param y: Label vector (-1 or +1)
    :return: w (weight vector), b (bias)
    """

    # Convert input matrix x into NumPy array
    X = np.array(x)
    
    # Compute prior probabilities P(Y)
    Y_pos, Y_neg = naivebayesPY(X, y)

    # Compute conditional probabilities P(X | Y)
    PXY_pos, PXY_neg = naivebayesPXY(X, y)

    # Ensure probabilities are valid to avoid division errors
    PXY_pos = np.clip(PXY_pos, 1e-10, 1)  # Avoid log(0)
    PXY_neg = np.clip(PXY_neg, 1e-10, 1)  # Avoid log(0)

    # Compute weight vector w and bias b using log-likelihood ratio
    w = np.log(PXY_pos) - np.log(PXY_neg)
    b = np.log(Y_pos) - np.log(Y_neg)

    return w, b
