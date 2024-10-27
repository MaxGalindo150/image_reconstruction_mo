import numpy as np
from pymoo.core.problem import Problem


class ImageReconstructionProblem(Problem):
    def __init__(self, MODEL, PROBE, SIGNAL, n_var=None, b=None, tikhonov_aprox=None):
        self.b = b
        self.MODEL = MODEL 
        self.PROBE = PROBE
        self.SIGNAL = SIGNAL
        self.n_var = 20 if n_var is None else n_var
        self.n_obj = 2
        if tikhonov_aprox is not None:
            tikhonov_aprox = tikhonov_aprox.flatten()  # Asegurarse de que sea un vector de una dimensión
            xl = tikhonov_aprox-1e-3
            xu = tikhonov_aprox+1e-3
        else:
            xl = 8000
            xu = 20000
        Problem.__init__(self, n_var=self.n_var, n_obj=self.n_obj, n_constr=0, xl=xl, xu=xu)
        
    def f1(self, x):
        if self.b is not None:
            res = self.b - self.MODEL.H @ x
        else:
            res = self.SIGNAL.y - self.MODEL.H @ x
        squared_error = np.sum(np.abs(res) ** 2)
        return squared_error

    def f2(self, x):
        # Penalizar la variación entre valores sucesivos en x
        diffs = np.diff(x, axis=0)  # Calcular las diferencias sucesivas
        variation_penalty = np.sum(diffs ** 2)  # Minimizar la suma de los cuadrados de las diferencias
        return variation_penalty

    def _evaluate(self, x, out, *args, **kwargs):
        f1_values = np.apply_along_axis(self.f1, 1, x)
        f2_values = np.apply_along_axis(self.f2, 1, x)
        out["F"] = np.column_stack([f1_values, f2_values])
    
    def mo_estimation(self, d_est):
        mu_est_mo = np.zeros(self.MODEL.Nd)
        vec_a_est_mo = np.zeros(self.MODEL.Nd)

        mu_est_mo[0] = d_est[0]

        for iter in range(1, self.MODEL.Nd):
            vec_a_est_mo[iter - 1] = np.exp(-mu_est_mo[iter - 1] * self.MODEL.dz)
            mu_est_mo[iter] = d_est[iter] / np.prod(vec_a_est_mo[:iter])

        return mu_est_mo
    
    