import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    numUnhappy = 0 # number of classes that didnt meet the desired margin
    for j in xrange(num_classes): # cycle through each class score
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if (j != y[i]) and (margin > 0):
        loss += margin
        numUnhappy += 1
        dW[:,j] += X[i]
    dW[:,y[i]] += -1*numUnhappy*X[i]
    
  # Right now the gradient is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  dW /= num_train
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Add regularization to the gradient
  dW += 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  delta = 1.0
  num_train = X.shape[0]
  scores = W.T.dot(X.T)
  correct_scores = np.choose(y,scores)
  margins = np.maximum(0, scores - correct_scores + delta)
  margins[y,range(num_train)] = 0
  loss = np.sum(margins)/num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  num_classes = W.shape[1]
  numUnhappy = np.sum(margins>0,axis=0)
  A = np.zeros((num_train,num_classes)) # initialize weight matrix
  A[range(num_train),y] = -1*numUnhappy # set elements for correct classes
  A[np.nonzero(margins.T)] = 1 # set elements for incorrect classes, where margins are not met 
  dW = X.T.dot(A)/num_train + 2*reg*W  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
