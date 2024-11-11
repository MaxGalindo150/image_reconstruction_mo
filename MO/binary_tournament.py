import numpy as np

def binary_tournament(pop, P, **kwargs):
    """Realiza la selección por torneo binario"""
    n_tournaments, n_competitors = P.shape
    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")
    S = np.full(n_tournaments, -1, dtype=int)
    for i in range(n_tournaments):
        a, b = P[i]
        S[i] = a if pop[a].F[0] < pop[b].F[0] else b
    return S
