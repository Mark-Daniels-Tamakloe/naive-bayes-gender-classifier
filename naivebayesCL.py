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

    # Get P(Y) for both classes
    pos, neg = naivebayesPY(x, y)
    
    # Get P(X|Y) for all features
    posprob, negprob = naivebayesPXY(x, y)
    posprob = np.array(posprob).flatten()
    negprob = np.array(negprob).flatten()
    
    # For the weights, we use: w_i = log(P(X_i=1|Y=1)/P(X_i=0|Y=1)) - log(P(X_i=1|Y=-1)/P(X_i=0|Y=-1))
    w = np.log(posprob/(1-posprob)) - np.log(negprob/(1-negprob))
    #w = np.log(posprob) - np.log(negprob)
    
    # For the bias, we use: b = log(P(Y=1)/P(Y=-1)) and add the sum of the logs of the probabilities of the features being 0
    #b = np.log(pos) - np.log(neg)
    b = np.log(pos) - np.log(neg) + np.log(1-posprob).sum() - np.log(1-negprob).sum()

    # Reshape w to be a column vector
    w = np.array(w).reshape(-1, 1)
    # print(w.shape)
    
    return w, b
# =============================================================================
