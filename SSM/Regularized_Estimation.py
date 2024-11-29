from types import SimpleNamespace

import numpy as np
from External_lib.Tikhonov import tikhonov
from External_lib.L_Curve import l_curve, plot_lc
# from External_lib.Tikhonov import tikhonov
from SSM.mu_from_d import mu_from_d


def regularized_estimation(
    model: SimpleNamespace, signal: SimpleNamespace, dim: int
) :
    """
    This function estimates the vector d as well as the absorption profile
    by using normal Tikhonov regularization.
    """

    H = model.H
    b = signal.y
    U, s, V = np.linalg.svd(H, full_matrices=False)
    L = np.eye(model.H.size)
    
    # This reshaping is necessary for the L curve method to have a 2D array (n,1)
    s = s.reshape(-1, 1)

    # L curve method
    rho, eta, reg_corner_tikh, rho_c, eta_c, reg_param = l_curve(U, s, b, "Tikh", L, V)
    d_est_tikh = tikhonov(H, b, reg_corner_tikh, dim=dim)
    plot_lc(rho, eta, reg_corner_tikh, rho_c, eta_c)

    return d_est_tikh, reg_corner_tikh


