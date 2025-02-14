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

    # Ensure w is (d,1) by reshaping
    w = np.log(PXY_pos / PXY_neg).reshape(-1, 1)  # Ensures correct shape
    
    # Ensure b is a scalar
    b = float(np.log(Y_pos) - np.log(Y_neg))  # Converts array to scalar

    return w, b
