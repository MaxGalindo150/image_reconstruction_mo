import numpy as np
from types import SimpleNamespace
def mo_estimation(d_est: np.array, model: SimpleNamespace) -> np.array:
    mu_est = np.zeros(d_est.shape)
    vec_a_est = np.zeros(d_est.shape)
    mu_est[0] = d_est[0]
    
    for iter in range(1, len(d_est)):
        vec_a_est[iter - 1] = np.exp(-mu_est[iter - 1] * model.dz)
        mu_est[iter] = d_est[iter] / np.prod(vec_a_est[:iter])
    return mu_est
