import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from SSM.Generate_Linear_Model import generate_linear_model
from SSM.Generate_Measurements import generate_measurements
from SSM.Generate_SSM_Model import generate_ssm_model
from SSM.Set_Settings import set_settings
from SSM.Regularized_Estimation import regularized_estimation
from MO.NSGA_II import NSGA2
import numpy as np
from pymoo.core.evaluator import Evaluator
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.ux import UniformCrossover


from MO.MOP_Definition import ImageReconstructionProblem
from MO.custom_crossover import CustomCrossover
from MO.custom_mutation import CustomMutation

from pymoo.core.problem import StarmapParallelization

import multiprocessing
from multiprocessing.pool import ThreadPool  

# Configuración del problema
model, PROBE, signal = set_settings(example=1)
model = generate_ssm_model(model, PROBE)
MODEL = generate_linear_model(model, signal, PROBE)
SIGNAL = generate_measurements(signal, MODEL, sigma=0)
n_var = 20
tikhonov_aprox = regularized_estimation(MODEL, SIGNAL, dim=1).reshape(n_var)  # Asegurarse de que sea un vector de una dimensión


problem = ImageReconstructionProblem(MODEL, PROBE, SIGNAL, n_var, tikhonov_aprox=None)

nsga2 = NSGA2(generations=450,population_size=100, mutation_rate=0.1, problem=problem)

x = [individual.values[0] for individual in nsga2.P_t]
y = [individual.values[1] for individual in nsga2.P_t]

best_individual_nsga2 = min(nsga2.P_t, key=lambda p: p.values[0])
best_solution = best_individual_nsga2.point


if np.allclose(best_solution, tikhonov_aprox, atol=1e-10):
    print('The best solution is the Tikhonov approximation')

mu_est_nsga2 = problem.mo_estimation(best_solution.reshape(20, 1))
mu_est_tikh = problem.mo_estimation(tikhonov_aprox.reshape(20, 1))

print(f'nsga: {mu_est_nsga2}')
print(f'tikh: {mu_est_tikh}')

plt.figure(figsize=(10, 6))  # Ajustar el tamaño de la figura
plt.plot(mu_est_nsga2, label='NSGA-II Estimation', linestyle='-', color='b')
plt.plot(PROBE.mu, label='True mu', linestyle='--', color='g')
plt.plot(mu_est_tikh, label='Tikhonov Estimation', linestyle='-.', color='r')

plt.xlabel('Depth', fontsize=14)
plt.ylabel('mu', fontsize=14)
plt.title('Sin Informar a NSGA-II (me)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.savefig('img/nsga2_pro/mu_est_nsga2_me.png')
plt.close()

plt.figure()
plt.scatter(x, y, c='red')
plt.title('Pareto Front (me)')
plt.xlabel('f1')
plt.ylabel('f2')
plt.savefig('pareto_front_me.png')

# # Visualizar el frente de Pareto
# Scatter().add(res.F).show().save('pareto_front.png')
