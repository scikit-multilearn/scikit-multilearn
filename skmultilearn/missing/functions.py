import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import inv

def label_correlation(y, s):
    """Correlation between labels in a label matrix

    Parameters
    ----------
    y : array-like (n_labels, n_samples)
        Label matrix

    s : float
        Smoothness parameter

    Returns
    -------
    L : array-like (n_labels, n_labels)
        Label correlation matrix

    """
    L = np.zeros(shape=[y.shape[0], y.shape[0]])

    for i in range(0, y.shape[0]):
        yi = sum(y[i,:])
        for j in range(0, y.shape[0]):
            coincidence = 0
            for k in range(0, y.shape[1]):
                if (int(y[i,k]) == int(1)) and (int(y[j,k]) == int(1)):
                    coincidence += 1
            L[i,j] = (coincidence + s)/(yi + 2*s)
    
    return L

def estimate_mising_labels(y, L):
    """Estimation of the missing labels, using the correlation matrix

    Parameters
    ----------
    y : array-like (n_labels, n_samples)
        Label matrix
    L : array-like (n_labels, n_labels)
        Label correlation matrix

    Returns
    -------
    estimate_matrix : array-like (n_labels, n_samples)
        Label estimation matrix
        y~ic = yiT * L(.,c) if yic == 0
        y~ic = 1 otherwise
    """
    estimate_matrix = np.zeros(shape=[y.shape[0],y.shape[1]], dtype=float)
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if int(y[j,i]) == int(0):
                aux = np.dot(np.transpose(y[:, i]), L[:,j])
                estimate_matrix[j,i] = aux
            else:
                estimate_matrix[j,i] = 1
    #Once we have the matrix, normalize the data
    estimate_matrix_copy = np.copy(estimate_matrix)
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if int(y[j,i]) == int(0):
                if np.sum(estimate_matrix_copy[:,i]) != 0:
                    aux = estimate_matrix_copy[j,i]/(np.sum(estimate_matrix_copy[:,i]))
                    estimate_matrix[j,i] = aux
            
    return estimate_matrix

def weight_adjacent_matrix(X, k):
    """Using the kNN algorithm we will use the clusters to get a weight matrix

    Parameters
    ----------
    X : array-like or sparse matrix (n_features, n_samples)
        Data to classify or in this case to make clusters
    k : int
        Number of clusters we want to make
    
    Returns
    -------
    W : array-like (n_samples, n_samples)
        Weighted matrix created from the predictions of kNN
        wij = 1 if xi is in the same cluster as xj
        wij = 0 other case
    """
    kNN = KMeans(n_clusters=k)
    kNN.fit(np.transpose(X))
    predictions = kNN.predict(np.transpose(X))
    W = np.zeros(shape=[X.shape[1], X.shape[1]], dtype=int)
    for i in range(0, X.shape[1]):
        for j in range(0, X.shape[1]):
            if int(predictions[i]) == int(predictions[j]):
                W[i,j] = 1
            else:
                W[i,j] = 0

    return W

def diagonal_matrix_H(X, y):
    """Diagonal matrix that indicates if X is labeled

    Parameters
    ----------
    X : array-like or sparse matrix (n_features, n_samples)
        Data to classify
    y : array-like (n_labels, n_samples)
        Labels of the data

    Returns
    -------
    H : array-like (n_samples, n_samples)
        Diagonal matrix indicating if an element of X is labeled or not
    """

    H = np.zeros(shape=[X.shape[1], X.shape[1]])

    for i in range(0, X.shape[1]):
        if np.sum(y[:, i]) > 0:
            H[i,i] = 1

    return H

def diagonal_matrix_lambda(W):
    """

    Parameters
    ----------
    W : array-like (n_samples, n_samples)
        Weighted matrix

    Returns
    -------
    diagonal_lambda : array-like (n_samples, n_samples)
        Diagonal matrix having the sum of weights of the weighted matrix
    """
    diagonal_lambda = np.zeros(shape=[W.shape[0], W.shape[1]])
    for i in range(0, W.shape[0]):
        diagonal_lambda[i,i] = np.sum(W[i,:])
    
    return diagonal_lambda

def graph_laplacian_matrix(lambda_matrix, W):
    """

    Parameters
    ----------
    lambda_matrix : array-like (n_samples, n_samples)
        Diagonal matrix having the sum of weights of the weighted matrix
    W : array-like (n_samples, n_samples)
        Weighted matrix

    Returns
    -------
    M : array-like (n_samples, n_samples)
        Graph laplacian matrix
    """
    M = np.zeros(shape=[W.shape[0], W.shape[1]])
    M = np.subtract(lambda_matrix, W)
    return M

def diagonal_matrix_Hc(H):
    """

    Parameters
    ----------
    H : array-like (n_samples, n_samples)
        Diagonal matrix indicating if an element of X is labeled or not

    Returns
    -------
    Hc : array-like (n_samples, n_samples)
        Hc = H - (H*1*1t*Ht)/(N)
    """
    Hc = np.zeros(shape = [H.shape[0], H.shape[0]])
    oneVector = np.ones(shape=[H.shape[0],1])
    numerator1 = np.matmul(H, oneVector)
    numerator2 = np.matmul(np.transpose(oneVector), np.transpose(H))
    numerator = np.matmul(numerator1, numerator2)
    product = numerator/H.shape[0]
    Hc = np.subtract(H, product)
    return Hc

def predictive_matrix(X, Hc, M, estimate_matrix, alpha):
    """Predictive matrix that works as the first item of the equation

    Parameters
    ----------
    X : array-like or sparse matrix (n_features, n_samples)
        Data to be classified or trained
    Hc : array-like (n_samples, n_samples)
        Diagonal matrix obtained from H
    M : array-like(n_samples, n_samples)
        Graph laplacian matrix

    Returns
    -------
    P : array-like (n_features, n_labels)
        P = (X*Hc*Xt + alpha*X*M*Xt)-1 * X*Hc*YPred
        R = dxc
    """
    P = np.zeros(shape=[X.shape[0], estimate_matrix.shape[0]])
    numerator1 = np.matmul(X, Hc)
    numerator1 = np.matmul(numerator1, np.transpose(X))
    numerator2 = np.matmul(M, np.transpose(X))
    numerator2 = alpha * np.matmul(X, numerator2)
    numerator = np.add(numerator1, numerator2)
    numerator = inv(numerator)
    numerator2 = np.matmul(X, Hc)
    numerator2 = np.matmul(numerator2, np.transpose(estimate_matrix))
    P = np.matmul(numerator, numerator2)

    return P

def label_bias(estimate_matrix, P, X, H):
    """Label bias that works as the second item of the equation

    Parameters
    ----------
    estimate_matrix : array-like (n_samples, n_samples)
        Diagonal matrix indicating if an element of X is labeled or not
    P : array-like (n_features, n_labels)
        Predictive item
    X : array-like (n_features, n_samples)
        Data to train or test
    H : array-like (n_samples, n_samples)
        Diagonal matrix indicating if an element of X is labeled or not

    Returns
    -------
    b : array-like (n_labels)
        Label bias as the second item of the equation
        b = ((estimate_matrix - Pt*X)*H*1)/N
    """
    b = np.zeros(estimate_matrix.shape[1])
    aux = np.matmul(np.transpose(P), X)
    numerator1 = np.subtract(estimate_matrix, aux)
    oneVector = np.ones(shape=[H.shape[0], 1])
    numerator2 = np.matmul(H, oneVector)
    numerator = np.matmul(numerator1, numerator2)
    b = numerator / H.shape[0]
    return b