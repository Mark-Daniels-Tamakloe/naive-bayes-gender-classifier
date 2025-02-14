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
    # =============================================================================
    # function [w,b]=naivebayesCL(x,y);
    #
    # Implementation of a Naive Bayes classifier
    # Input:
    # x : n input vectors of d dimensions (dxn)
    # y : n labels (-1 or +1)
    #
    # Output:
    # w : weight vector
    # b : bias (scalar)
    # =============================================================================

    # Converting input matrix x and y into NumPy matrix
    # Input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    Y = np.matrix(y)

    # Pre-configuring the size of matrix X
    d, n = X.shape

    # =============================================================================
    # Fill in code here
    py_plus, py_minus = naivebayesPY(X, Y)
    pxy_plus, pxy_minus = naivebayesPXY(X, Y)

    w = np.log(pxy_plus / pxy_minus)
    b = np.log(py_plus / py_minus)

    return w, b
    # =============================================================================
