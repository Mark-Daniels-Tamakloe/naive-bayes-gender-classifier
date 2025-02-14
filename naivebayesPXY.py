#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
@author: Yichen
@author: M.Joo (smoothing with all zeros)
"""

import numpy as np

def naivebayesPXY(x, y):
    """
    Compute P(X|Y) using a Bernoulli Naïve Bayes assumption.
    
    :param x: Feature matrix (dxn)
    :param y: Label vector (-1 or +1) (1xn)
    :return: posprob (dx1), negprob (dx1) 
             where each entry is P(X_alpha = 1 | Y = y)
    """

    # Convert input matrix x and y into NumPy arrays
    X = np.array(x)  # Ensure X is a NumPy array
    Y = np.array(y).reshape(-1)  # Ensure Y is a 1D array

    d, n = X.shape  # Get dimensions

    # Laplace Smoothing Parameter
    smoothing = 1  # Add-one Laplace smoothing

    # Create masks for Y = +1 and Y = -1
    pos_mask = (Y == 1)  # Boolean array where y = +1
    neg_mask = (Y == -1) # Boolean array where y = -1

    # Compute total counts for y = +1 and y = -1
    total_pos = np.sum(pos_mask)  # Count number of positive samples
    total_neg = np.sum(neg_mask)  # Count number of negative samples

    # Prevent division by zero
    total_pos = max(total_pos, 1)
    total_neg = max(total_neg, 1)

    # Compute feature-wise counts given y = +1 and y = -1 with smoothing
    pos_counts = np.sum(X[:, pos_mask], axis=1, keepdims=True) + smoothing  
    neg_counts = np.sum(X[:, neg_mask], axis=1, keepdims=True) + smoothing  

    # Compute probabilities
    posprob = pos_counts / (total_pos + 2)  # Normalizing by class count
    negprob = neg_counts / (total_neg + 2)

    return posprob, negprob
