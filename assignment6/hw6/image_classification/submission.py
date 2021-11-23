import collections
import numpy as np

############################################################
# Problem 4.1

def runKMeans(k,patches,maxIter):
    """
    Runs K-means to learn k centroids, for maxIter iterations.
    
    Args:
      k - number of centroids.
      patches - 2D numpy array of size patchSize x numPatches
      maxIter - number of iterations to run K-means for

    Returns:
      centroids - 2D numpy array of size patchSize x k
    """
    # This line starts you out with randomly initialized centroids in a matrix 
    # with patchSize rows and k columns. Each column is a centroid.
    patchSize = patches.shape[0]
    numPatches = patches.shape[1]
    centroids = np.random.randn(patchSize, k)

    for iteration in range(maxIter):
        # BEGIN_YOUR_CODE (around 19 lines of code expected)

        # Assignment step
        z = np.zeros(numPatches)
        for i in range(numPatches): # assign a zi to each xi
            min_dist = math.inf
            xi = patches[:,i]
            for j in range(k): # find the minimizing Uk
                uj = centroids[:,j]
                diff_ij = xi-uj
                dist_ij = np.sum(diff_ij**2)
                if dist_ij < min_dist:
                    min_dist = dist_ij
                    z[i] = j

        # Update step
        for i in range(k): # loop over all centroids
            vector_sum = np.zeros(patchSize)
            num_vectors = 0
            for j in range(numPatches): # loop over all elements of z
                if z[j] == i:
                    num_vectors = num_vectors + 1
                    xj = patches[:, j]
                    vector_sum = vector_sum + xj
            uk_new = vector_sum/num_vectors
            centroids[:,i] = uk_new

        # END_YOUR_CODE

    return centroids

############################################################
# Problem 4.2

def extractFeatures(patches,centroids):
    """
    Given patches for an image and a set of centroids, extracts and return
    the features for that image.
    
    Args:
      patches - 2D numpy array of size patchSize x numPatches
      centroids - 2D numpy array of size patchSize x k
      
    Returns:
      features - 2D numpy array with new feature values for each patch
                 of the image in rows, size is numPatches x k
    """
    k = centroids.shape[1]
    numPatches = patches.shape[1]
    features = np.empty((numPatches,k))

    # BEGIN_YOUR_CODE (around 9 lines of code expected)
    for i in range(numPatches):
        xi = patches[:,i]
        for k_index in range(k):
            uk = centroids[:,k_index]
            norm_sum = 0
            for j in range(k):
                uj = centroids[:,j]
                norm_ij = math.sqrt(np.sum((xi-uj)**2))
                norm_sum = norm_sum + norm_ij
            a_ijk = norm_sum/k - math.sqrt(np.sum((xi-uk)**2))
            features[i,k_index] = max(a_ijk, 0)

    # END_YOUR_CODE
    return features

############################################################
# Problem 4.3.1

import math
def logisticGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of the logistic loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of logistic loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    yy = 2*y - 1
    c = -yy*np.exp(-yy*np.dot(theta, featureVector))/(1 + np.exp(-yy*np.dot(theta, featureVector)))
    return c*featureVector
    # END_YOUR_CODE

############################################################
# Problem 4.3.2
    
def hingeLossGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of hinge loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of hinge loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    yy = 2*y - 1
    if 1-np.dot(theta, featureVector)*yy > 0:
        return -yy*featureVector
    else:
        return 0*featureVector
    # END_YOUR_CODE

