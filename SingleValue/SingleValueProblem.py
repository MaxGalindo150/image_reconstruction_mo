import numpy as np
from pymoo.core.problem import Problem


from External_lib.Tikhonov import tikhonov

class SingleValueReconstructionProblem(Problem):
    def __init__(self, MODEL, SIGNAL, l_curve_sol=None):
        self.H = MODEL.H  # Matriz del modelo
        self.y = SIGNAL.y  # Señal de medición
        self.MODEL = MODEL  # Modelo
        self.f1_max, self.f2_max, self.f3_max = 1, 1, 1  # Inicialización para normalización
        self.f1_min, self.f2_min, self.f3_min = 0, 0, 0  # Para normalización min-max

        if l_curve_sol is not None:
            xl = l_curve_sol - 1e-3
            xu = l_curve_sol + 1e-3
        else:
            xl = 0
            xu = 1
        
        super().__init__(n_var=1, n_obj=3, n_constr=0, xl=xl, xu=xu)

    def f1(self, d_hat):
        # Objetivo 1: Residual
        residual = np.sum((self.y - self.H @ d_hat) ** 2, axis=0)
        return residual
        
    def f2(self, d_hat):
        # Objetivo 2: Regularización
        regularization = np.sum(d_hat ** 2)
        return regularization

    def f3(self, d_hat):
        # Objetivo 3: Penalización de valores negativos
        negativity_penalty = np.sum(np.abs(d_hat[d_hat < 0]))
        return negativity_penalty

    def _evaluate(self, x, out, *args, **kwargs):
        lambdas = x.flatten()  # Asegúrate de que x sea un vector plano

        # Inicializar resultados
        residuals = []
        regularizations = []
        negativities = []

        for lambd in lambdas:
            # Calcular d_hat para cada lambda
            d_hat = tikhonov(self.H, self.y, lambd, dim=1)
            #d_hat = np.linalg.inv(self.H.T @ self.H + lambd * np.eye(self.H.shape[1])) @ self.H.T @ self.y

            # Calcular objetivos
            residuals.append(self.f1(d_hat))
            regularizations.append(self.f2(d_hat))
            negativities.append(self.f3(d_hat))

        # Actualizar máximos y mínimos para normalización
        self.f1_max, self.f1_min = max(residuals), min(residuals)
        self.f2_max, self.f2_min = max(regularizations), min(regularizations)
        self.f3_max, self.f3_min = max(negativities), min(negativities)

        # Normalizar objetivos (min-max normalization)
        residuals_norm = [(r - self.f1_min) / (self.f1_max - self.f1_min) if self.f1_max != self.f1_min else r for r in residuals]
        regularizations_norm = [(r - self.f2_min) / (self.f2_max - self.f2_min) if self.f2_max != self.f2_min else r for r in regularizations]
        negativities_norm = [(n - self.f3_min) / (self.f3_max - self.f3_min) if self.f3_max != self.f3_min else n for n in negativities]

        # Salida de los objetivos normalizados
        out["F"] = np.column_stack([residuals_norm, regularizations_norm, negativities_norm])

    def evaluate_l_curve_solution(self, l_curve_sol):
        """
        Evalúa la solución de la L-Curve en términos de los objetivos normalizados.

        Args:
            l_curve_sol (float): Valor de lambda correspondiente a la L-Curve.

        Returns:
            ndarray: Valores normalizados de los objetivos para la solución L-Curve.
        """
        # Calcular d_hat para la solución de L-Curve
        d_hat = tikhonov(self.H, self.y, l_curve_sol, dim=1)

        # Evaluar objetivos sin normalizar
        f1 = self.f1(d_hat)
        f2 = self.f2(d_hat)
        f3 = self.f3(d_hat)

        # Actualizar los límites máximos y mínimos si es necesario
        self.f1_max = max(f1, self.f1_max)
        self.f1_min = min(f1, self.f1_min)
        self.f2_max = max(f2, self.f2_max)
        self.f2_min = min(f2, self.f2_min)
        self.f3_max = max(f3, self.f3_max)
        self.f3_min = min(f3, self.f3_min)

        # Normalizar los objetivos
        f1_norm = (f1 - self.f1_min) / (self.f1_max - self.f1_min) if self.f1_max != self.f1_min else f1
        f2_norm = (f2 - self.f2_min) / (self.f2_max - self.f2_min) if self.f2_max != self.f2_min else f2
        f3_norm = (f3 - self.f3_min) / (self.f3_max - self.f3_min) if self.f3_max != self.f3_min else f3

        return np.array([f1_norm[0], f2_norm, f3_norm])

    def mo_estimation(self, d_est):
        """
        Estima mu usando el vector de decisión estimado.
        """
        mu_est_mo = np.zeros(self.MODEL.Nd)
        vec_a_est_mo = np.zeros(self.MODEL.Nd)

        mu_est_mo[0] = d_est[0]

        for iter in range(1, self.MODEL.Nd):
            vec_a_est_mo[iter - 1] = np.exp(-mu_est_mo[iter - 1] * self.MODEL.dz)
            mu_est_mo[iter] = d_est[iter] / np.prod(vec_a_est_mo[:iter])

        return mu_est_mo
    
    def select_best_solution(self, F, weights=None):
        """
        Select the best solution from the Pareto front based on a weighted score.

        Args:
            F (ndarray): The objective values of the solutions in the Pareto front, normalized.
            weights (list or ndarray): Weights for each objective. Should sum to 1. Defaults to equal weights.

        Returns:
            int: Index of the best solution in the Pareto front.
            ndarray: The best solution in terms of the objectives.
        """
        #Set default weights to equal importance if not provided
        if weights is None:
            weights = np.ones(F.shape[1]) / F.shape[1]

        # Ensure weights are normalized
        weights = np.array(weights)
        if not np.isclose(np.sum(weights), 1):
            weights /= np.sum(weights)

        # Calculate the weighted score for each solution
        scores = np.dot(F, weights)

        # Find the index of the solution with the minimum score
        best_index = np.argmin(scores)

        # Return the index and the corresponding objective values
        return best_index, F[best_index]
        

        # min_index = np.argmin(F[:, 0])

        
        # return min_index, F[min_index]