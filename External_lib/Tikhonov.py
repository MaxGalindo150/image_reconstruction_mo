import numpy as np

def tikhonov(H, b,lambdas):
    """
    This function computes the Tikhonov regularization.
    :param H: matrix of the linear model
    :param b: measurements
    :param lambdas: regularization parameter
    :return: estimated signal
    """

    I = np.eye(H.shape[1])
    # Compute the Tikhonov regularization
    d_est_LSQ = np.linalg.inv(H.T @ H + 0.00001*lambdas[0]*I) @ H.T @ b
    return d_est_LSQ
