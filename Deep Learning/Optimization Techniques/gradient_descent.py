import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    m,n = X.shape
    if method == "batch":
        method_batch_size = m
    elif method == "stochastic":
        method_batch_size = 1
    else:
        method_batch_size = batch_size
    
    for epoch in range(n_epochs):
        for i in range(0, m, method_batch_size):
            X_batch = X[i:i+method_batch_size]
            y_batch = y[i:i+method_batch_size]

            predictions = X_batch @ weights
            error = predictions - y_batch

            m_batch = X_batch.shape[0]
            gradient = 2/m_batch*(X_batch.T @ error)

            weights = weights - (learning_rate * gradient)
    
    return weights

