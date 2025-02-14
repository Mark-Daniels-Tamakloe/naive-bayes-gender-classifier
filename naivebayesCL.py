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
    Implementation of a Naive Bayes classifier
    Input:
        x : n input vectors of d dimensions (dxn)
        y : n labels (-1 or +1)
    Output:
        w : weight vector
        b : bias (scalar)
    """

    # Ensure X is a NumPy array (avoid np.matrix)
    X = np.array(x)

    # Get class priors P(Y=+1) and P(Y=-1)
    Y_pos, Y_neg = naivebayesPY(X, y)

    # Compute conditional probabilities P(X|Y)
    PXY_pos, PXY_neg = naivebayesPXY(X, y)

    # Ensure probabilities are column vectors
    PXY_pos = PXY_pos.reshape(-1, 1)
    PXY_neg = PXY_neg.reshape(-1, 1)

    # Compute weight and bias
    w = np.log(PXY_pos / PXY_neg)
    b = np.log(Y_pos / Y_neg)

    return w, b
