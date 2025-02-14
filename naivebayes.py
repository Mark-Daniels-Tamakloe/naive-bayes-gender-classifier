#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY

def naivebayes(x, y, x1):
    # =============================================================================
    # function logratio = naivebayes(x,y,x1);
    #
    # Computation of log P(Y|X=x1) using Bayes Rule
    # Input:
    # x : n input vectors of d dimensions (dxn)
    # y : n labels (-1 or +1)
    # x1: input vector of d dimensions (dx1)
    #
    # Output:
    # logratio: log (P(Y = 1|X=x1)/P(Y=-1|X=x1))
    # =============================================================================

    # Convert input matrix x and x1 into NumPy arrays
    X = np.array(x)
    X1 = np.array(x1).reshape(-1, 1)  # Ensure column vector

    # Get prior probabilities
    pos, neg = naivebayesPY(x, y)

    # Get conditional probabilities P(X|Y)
    posprob, negprob = naivebayesPXY(x, y)

    # Compute likelihoods P(X1|Y=1) and P(X1|Y=-1)
    likelihood_pos = np.prod(posprob ** X1 * (1 - posprob) ** (1 - X1), axis=0)
    likelihood_neg = np.prod(negprob ** X1 * (1 - negprob) ** (1 - X1), axis=0)

    # Compute posterior probabilities
    P_Y1_X = likelihood_pos * pos
    P_Yneg1_X = likelihood_neg * neg

    # Compute log ratio
    logratio = np.log(P_Y1_X) - np.log(P_Yneg1_X)

    return logratio
