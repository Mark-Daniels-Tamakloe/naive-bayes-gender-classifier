#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np

def naivebayesPY(x, y):
    # function [pos,neg] = naivebayesPY(x,y);
    #
    # Computation of P(Y)
    # Input:
    # x : n input vectors of d dimensions (dxn)
    # y : n labels (-1 or +1) (1xn)
    #
    # Output:
    # pos: probability p(y=1)
    # neg: probability p(y=-1)
    #
    
    # Convertng input matrix x and y into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    Y = np.matrix(y)
    
    # Pre-configuring the size of matrix X
    d,n = X.shape
    
    # Pre-constructing a matrix of all-ones (dx2)
    X0 = np.ones((d,2))
    Y0 = np.matrix('-1, 1')
    
    # add one all-ones positive and negative example
    Xnew = np.hstack((X, X0)) #stack arrays in sequence horizontally (column-wise)
        #Xnew = np.concatenate((X, X0), axis=1) #concatenate to column
    Ynew = np.hstack((Y, Y0))
    # Re-configuring the size of matrix Xnew
    d,n = Xnew.shape
    
    ## fill in code here
    # YOUR CODE HERE

    # Count the occurrences of each class
    pos_count = np.sum(Ynew == 1)  # Number of positive examples
    neg_count = np.sum(Ynew == -1) # Number of negative examples
    total_count = pos_count + neg_count  # Total number of examples

    # Compute class probabilities
    pos = pos_count / total_count  # P(Y=1)
    neg = neg_count / total_count  # P(Y=-1)

    return pos,neg