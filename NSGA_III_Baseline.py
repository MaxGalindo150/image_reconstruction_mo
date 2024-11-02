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
import numpy as np
from pymoo.core.evaluator import Evaluator


from MO.MOP_Definition_Pymoo import ImageReconstructionProblem
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

# Crear las direcciones de referencia para la optimización
ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)

# Crear el objeto del algoritmo
algorithm = NSGA3(pop_size=500, ref_dirs=ref_dirs)



# Ejecutar la optimización
res = minimize(problem,
               algorithm,
               seed=1,
               termination=('n_gen',1000),
               verbose=True)  # Asignar el número de procesos aquí

# Acceder a las soluciones y valores de las funciones objetivo
solutions = res.X
objective_values = res.F

# Encontrar el índice de la solución con el menor valor en el objetivo 1
min_index = np.argmin(objective_values[:, 0])

# Obtener la solución correspondiente a ese índice
best_solution = solutions[min_index]
best_objective_values = objective_values[min_index]

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
plt.title('Comparison of mu Estimations', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.savefig('img/nsga2_pro/mu_est_nsga2.png')
plt.close()

# Visualizar el frente de Pareto
Scatter().add(res.F).show().save('pareto_front.png')
