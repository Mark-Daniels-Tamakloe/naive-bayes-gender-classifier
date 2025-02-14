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

    # Convert input matrix x into NumPy array
    X = np.array(x)

    # Compute prior probabilities
    pos, neg = naivebayesPY(x, y)

    # Compute conditional probabilities P(X|Y)
    posprob, negprob = naivebayesPXY(x, y)

    # Compute weights (w)
    w = np.log(posprob) - np.log(negprob)

    # Compute bias (b)
    b = np.log(pos) - np.log(neg)

    return w, b
