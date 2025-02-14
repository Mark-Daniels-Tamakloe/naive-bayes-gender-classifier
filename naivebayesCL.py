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
    Compute Naïve Bayes classifier parameters (weights w and bias b).
    
    :param x: Feature matrix (dxn)
    :param y: Label vector (-1 or +1) (1xn)
    :return: w (dx1) weight vector, b (scalar bias)
    """

    # Convert input to NumPy array
    X = np.array(x)

    # Compute P(Y)
    Y_pos, Y_neg = naivebayesPY(X, y)

    # Compute P(X|Y)
    PXY_pos, PXY_neg = naivebayesPXY(X, y)

    # Compute weight vector w
    w = np.log(PXY_pos) - np.log(PXY_neg)  # Ensures correct element-wise logarithm
    
    # Compute bias b
    b = np.log(Y_pos) - np.log(Y_neg)  # Log of prior probabilities

    return w, b
