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
        Problem.__init__(self, n_var=self.n_var, n_obj=self.n_obj, n_constr=0, xl=tikhonov_aprox-1e-10, xu=tikhonov_aprox+1e-10)
        
    def f1(self, x):
        if self.b is not None:
            res = self.b - self.MODEL.H @ x
        else:
            res = self.SIGNAL.y - self.MODEL.H @ x
        # magnitude = np.linalg.norm(res, ord=2)
        squared = np.sum(res**2)
        return squared
    
    def f2(self, x):
        return -np.linalg.norm(x, ord=2)

    
    def evaluate(self, x):
        x = x.reshape(self.n_var, 1)
        return np.array([self.f1(x), self.f2(x)])
    
    def mo_estimation(self, d_est):
        mu_est_mo = np.zeros(self.MODEL.Nd)
        vec_a_est_mo = np.zeros(self.MODEL.Nd)

        mu_est_mo[0] = d_est[0]

        for iter in range(1, self.MODEL.Nd):
            vec_a_est_mo[iter - 1] = np.exp(-mu_est_mo[iter - 1] * self.MODEL.dz)
            mu_est_mo[iter] = d_est[iter] / np.prod(vec_a_est_mo[:iter])


        return mu_est_mo
        
