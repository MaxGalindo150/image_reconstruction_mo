from pymoo.core.crossover import Crossover
import numpy as np

class CustomCrossover(Crossover):
    def __init__(self, prob=0.9):
        # El crossover se aplica a dos padres y devuelve dos hijos
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        # X es una matriz de padres (n_parents x n_var)
        n_matings, n_var = X.shape[0] // 2, X.shape[1]
        Y = np.full_like(X, np.nan)

        for k in range(n_matings):
            if np.random.rand() < self.prob:
                p1, p2 = X[2 * k], X[2 * k + 1]

                # Cruce aritmético basado en tu implementación
                alpha = np.random.uniform(0, 1)
                new_point1 = alpha * p1 + (1 - alpha) * p2
                new_point2 = (1 - alpha) * p1 + alpha * p2

                Y[2 * k, :] = new_point1
                Y[2 * k + 1, :] = new_point2
            else:
                # Si no se realiza el cruce, se copian los padres
                Y[2 * k, :] = X[2 * k]
                Y[2 * k + 1, :] = X[2 * k + 1]

        return Y
