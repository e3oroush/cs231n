import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  num_classes=W.shape[1]
  f=X.dot(W) # N x C
  f-=f.max()
  f=np.exp(f)
  for i,fi in enumerate(f):
    f_sum=fi.sum()
    pi=fi[y[i]]/f_sum
    loss+=-np.log(pi)
    dW[:,y[i]] += (pi - 1)*X[i]
    for j in range(num_classes):
      if j == y[i]:
        continue
      pj=fi[j]/f_sum
      dW[:,j] += (pj)*X[i]
  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W*W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  f=X.dot(W) # N x C
  f-=f.max()
  f=np.exp(f)
  F=f/f.sum(axis=-1).reshape(-1,1) # N x C
  p=F[np.arange(num_train),y] # N x 1
  loss = np.sum(-np.log(p))/num_train
  loss += reg * np.sum(W*W)
  F[np.arange(num_train),y]-=1
  dW = X.T.dot(F)
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

