import numpy as np
import matplotlib.pyplot as plt
from External_lib.L_Corner import l_corner

def l_curve(U, sm, b, method, L, V):
    """
    This function computes the L curve method to find the regularization parameter
    for Tikhonov regularization.
    """
    # Initialize the variables
    n_points = 200
    smin_ratio = 16*np.finfo(float).eps
    
    m, n = U.shape
    p, ps = sm.shape

    beta = U.conj().T @ b
    beta2 = np.linalg.norm(b) ** 2 - np.linalg.norm(beta) ** 2

    if ps == 1:
        s = sm
        beta = beta[0:p]
    else:
        s = sm[p-1::-1,0] / sm[p-1::-1,0]
        beta = beta[p-1::-1]
     
    xi = beta[0:p] / s
    xi[np.isinf(xi)] = 0

    if method == "Tikh" or method == "tikh":

        # eta = ||x||^2
        eta = np.zeros((n_points,1))

        # rho = ||Ax - b||^2
        rho = np.zeros((n_points,1))

        # reg_param = regularization parameter
        reg_param = np.zeros((n_points,1))

        s2 = s ** 2
        
        reg_param[n_points-1] = max(s[p-1], s[0] * smin_ratio)
        ratio = (s[0] / reg_param[n_points-1]) ** (1 / (n_points - 1))
        
        for i in range(n_points-2,-1,-1):
            reg_param[i] = reg_param[i+1] * ratio

        for i in range(n_points):
            f = s2 / (s2 + reg_param[i] ** 2)

            eta[i] = np.linalg.norm(f * xi)

            rho[i] = np.linalg.norm((1 - f) * beta[0:p])

        if (m > n) and (beta2 > 0):
            rho = np.sqrt(rho ** 2 + beta2)

    
    reg_corner, rho_c, eta_c = l_corner(rho, eta, reg_param, U, s, b, "Tikh")
    print(f"(rho_c, eta_c): ({rho_c}, {eta_c})")
    return  rho, eta, reg_corner, rho_c, eta_c, reg_param


def plot_lc(rho, eta, reg_param, reg_corner, rho_c, eta_c):
    """
    This function plots the L curve and marks the corner found.
    """
    plt.figure()
    plt.plot(rho, eta, "o-", label="L curve")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("||Ax - b||")
    plt.ylabel("||x||")
    plt.title("L curve")
    plt.grid(True)
    
    # Marcar la esquina encontrada
    plt.plot(rho_c, eta_c, 'ro', label=f'Corner: {reg_corner}')
    
    # Anotar la esquina encontrada
    plt.annotate(f'Corner: {reg_corner}', (rho_c, eta_c),
                 textcoords="offset points", xytext=(10,-10), ha='center', color='red')
    
    plt.legend()
    plt.savefig("L_curve.png")
        
