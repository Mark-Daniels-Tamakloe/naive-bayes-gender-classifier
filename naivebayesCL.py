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
#function [w,b]=naivebayesCL(x,y);
#
#Implementation of a Naive Bayes classifier
#Input:
#x : n input vectors of d dimensions (dxn)
#y : n labels (-1 or +1)
#
#Output:
#w : weight vector
#b : bias (scalar)
# =============================================================================



    # Convertng input matrix x and x1 into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)

    # Pre-configuring the size of matrix X
    d,n = X.shape

# =============================================================================
# fill in code here
    # YOUR CODE HERE

    # Compute class prior probabilities
    pos, neg = naivebayesPY(x, y)

    # Compute conditional probabilities P(X|Y)
    posprob, negprob = naivebayesPXY(x, y)

    # Compute log-odds for linear classification
    w = np.log(posprob) - np.log(negprob)  # Weight vector
    b = np.log(pos) - np.log(neg) + np.sum(np.log(1 - negprob) - np.log(1 - posprob))  # Bias term

    return w,b
# =============================================================================
