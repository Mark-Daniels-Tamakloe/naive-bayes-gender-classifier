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

    # Compute prior probabilities P(Y)
    P_Ypos, P_Yneg = naivebayesPY(X, y)

    # Compute conditional probabilities P(X | Y)
    PXY_pos, PXY_neg = naivebayesPXY(X, y)

    # Compute weight vector w and bias b
    w = np.log(PXY_pos) - np.log(PXY_neg)  # Log probability ratio
    b = np.log(P_Ypos) - np.log(P_Yneg)    # Log prior ratio

    return w, b
