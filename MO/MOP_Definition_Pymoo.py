import numpy as np
from pymoo.core.problem import Problem

class ImageReconstructionProblem(Problem):
    def __init__(self, MODEL, PROBE, SIGNAL, n_var=None, b=None, tikhonov_aprox=None):
        self.b = b
        self.MODEL = MODEL 
        self.PROBE = PROBE
        self.SIGNAL = SIGNAL
        self.f1_max = None
        self.f2_max = None
        self.f3_max = None
        self.n_var = 20 if n_var is None else n_var
        self.n_obj = 2  # Cambiar a 3 si añades un tercer objetivo
        if tikhonov_aprox is not None:
            tikhonov_aprox = tikhonov_aprox.flatten()
            xl = tikhonov_aprox - 1000  # Asegurar positividad
            xu = tikhonov_aprox + 10
        else:
            xl = 750 # Establecer límite inferior en 0 para positividad
            xu = 25000
        Problem.__init__(self, n_var=self.n_var, n_obj=self.n_obj, n_constr=0, xl=xl, xu=xu)
        
    def f1(self, x):
        if self.b is not None:
            res = self.b - self.MODEL.H @ x
        else:
            res = self.SIGNAL.y - self.MODEL.H @ x
        squared_error = np.sum(np.abs(res) ** 2)
        return squared_error

    

    def f2(self, x):
        return -np.linalg.norm(x, ord=2)
    
    def f3(self, x):
        diffs = np.diff(x, axis=0)
        smoothness_penalty = np.sum(diffs ** 2)
        return smoothness_penalty
    
    # Si deseas añadir un tercer objetivo para la esparsidad:
    # def f3(self, x):
    #     sparsity_penalty = np.sum(np.abs(x))
    #     return sparsity_penalty

    def _evaluate(self, x, out, *args, **kwargs):
        f1_values = np.apply_along_axis(self.f1, 1, x)
        f2_values = np.apply_along_axis(self.f2, 1, x)
        f3_values = np.apply_along_axis(self.f3, 1, x)

        # Calcular y almacenar valores máximos
        self.f1_max = np.max(f1_values) if self.f1_max is None else self.f1_max
        self.f2_max = np.max(f2_values) if self.f2_max is None else self.f2_max
        self.f3_max = np.max(f3_values) if self.f3_max is None else self.f3_max

        # Normalización (si es necesario)
        f1_values_normalized = f1_values / self.f1_max if self.f1_max != 0 else f1_values
        f2_values_normalized = f2_values / self.f2_max if self.f2_max != 0 else f2_values
        f3_values_normalized = f3_values / self.f3_max if self.f3_max != 0 else f3_values

        out["F"] = np.column_stack([f1_values_normalized, f2_values_normalized])


    def mo_estimation(self, d_est):
        mu_est_mo = np.zeros(self.MODEL.Nd)
        vec_a_est_mo = np.zeros(self.MODEL.Nd)

        mu_est_mo[0] = d_est[0]

        for iter in range(1, self.MODEL.Nd):
            vec_a_est_mo[iter - 1] = np.exp(-mu_est_mo[iter - 1] * self.MODEL.dz)
            mu_est_mo[iter] = d_est[iter] / np.prod(vec_a_est_mo[:iter])

        return mu_est_mo
    
    # Evaluar la solución de Tikhonov
    def evaluate_tikhonv(self, tikhonov_solution):
        f1 = self.f1(tikhonov_solution)
        f2 = self.f2(tikhonov_solution)
        f3 = self.f3(tikhonov_solution) 

        f1_normalized = f1 / self.f1_max if self.f1_max != 0 else f1
        f2_normalized = f2 / self.f2_max if self.f2_max != 0 else f2
        f3_normalized = f3 / self.f3_max if self.f3_max != 0 else f3

        return [f1_normalized, f2_normalized]

