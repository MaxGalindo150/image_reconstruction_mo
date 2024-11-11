import numpy as np
from pymoo.core.problem import Problem

class GeneralReconstructionProblem(Problem):
    def __init__(self, MODEL, PROBE, SIGNAL, n_var=None, b=None, tikhonov_aprox=None):
        self.b = b
        self.MODEL = MODEL
        self.PROBE = PROBE
        self.SIGNAL = SIGNAL

        # Valores máximos para normalización
        self.f1_max = 1
        self.f2_max = 1
        self.f3_max = 1

        # Número de variables
        self.n_var = 20 if n_var is None else n_var

        # Número de objetivos (por defecto 2, puede extenderse a 3)
        self.n_obj = 2

        # Rango de las variables
        if tikhonov_aprox is not None:
            tikhonov_aprox = tikhonov_aprox.flatten()
            xl = tikhonov_aprox - 1000
            xu = tikhonov_aprox + 10
        else:
            xl = 750
            xu = 25000

        # Inicialización del problema
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=0, xl=xl, xu=xu)

    def f1(self, x):
        """
        Objetivo 1: Residual
        """
        res = (self.b - self.MODEL.H @ x) if self.b is not None else (self.SIGNAL.y - self.MODEL.H @ x)
        return np.sum(np.abs(res) ** 2)

    def f2(self, x):
        """
        Objetivo 2: Regularización
        """
        return np.linalg.norm(x, ord=2)

    def f3(self, x):
        """
        Objetivo 3: Suavidad (opcional)
        """
        diffs = np.diff(x, axis=0)
        return np.sum(diffs ** 2)

    def normalize_objectives(self, f1, f2, f3=None):
        """
        Normaliza los valores de los objetivos.
        """
        f1_normalized = f1 / self.f1_max if self.f1_max != 0 else f1
        f2_normalized = f2 / self.f2_max if self.f2_max != 0 else f2
        f3_normalized = f3 / self.f3_max if self.f3_max != 0 else f3 if f3 is not None else None
        return f1_normalized, f2_normalized, f3_normalized

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evalúa los objetivos para las soluciones dadas.
        """
        # Calcular objetivos
        f1_values = np.apply_along_axis(self.f1, 1, x)
        f2_values = np.apply_along_axis(self.f2, 1, x)
        f3_values = np.apply_along_axis(self.f3, 1, x) if self.n_obj == 3 else None

        # Calcular valores máximos
        self.f1_max = max(self.f1_max, np.max(f1_values))
        self.f2_max = max(self.f2_max, np.max(f2_values))
        if f3_values is not None:
            self.f3_max = max(self.f3_max, np.max(f3_values))

        # Normalización
        f1_norm, f2_norm, f3_norm = self.normalize_objectives(f1_values, f2_values, f3_values)

        # Salida de objetivos
        if self.n_obj == 2:
            out["F"] = np.column_stack([f1_norm, f2_norm])
        elif self.n_obj == 3:
            out["F"] = np.column_stack([f1_norm, f2_norm, f3_norm])

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

    def evaluate_tikhonov(self, tikhonov_solution):
        """
        Evalúa la solución de Tikhonov en relación a los objetivos.
        """
        f1 = self.f1(tikhonov_solution)
        f2 = self.f2(tikhonov_solution)
        f3 = self.f3(tikhonov_solution) if self.n_obj == 3 else None

        # Normalización
        f1_normalized, f2_normalized, f3_normalized = self.normalize_objectives(f1, f2, f3)

        if self.n_obj == 2:
            return [f1_normalized, f2_normalized]
        elif self.n_obj == 3:
            return [f1_normalized, f2_normalized, f3_normalized]
