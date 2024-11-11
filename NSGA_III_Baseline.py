import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS

from SSM.Generate_Linear_Model import generate_linear_model
from SSM.Generate_Measurements import generate_measurements
from SSM.Generate_SSM_Model import generate_ssm_model
from SSM.Set_Settings import set_settings
from SSM.Regularized_Estimation import regularized_estimation


import numpy as np

from MO.MOP_Definition_Pymoo import ImageReconstructionProblem
from MO.custom_crossover import CustomCrossover
from MO.custom_mutation import CustomMutation


# Configuración del problema
model, PROBE, signal = set_settings(example=1)
model = generate_ssm_model(model, PROBE)
MODEL = generate_linear_model(model, signal, PROBE)
SIGNAL = generate_measurements(signal, MODEL, sigma=0)

# Dimensión del vector a estimar
n_var = 20
tikhonov_aprox, _ = regularized_estimation(MODEL, SIGNAL, dim=1) # Asegurarse de que sea un vector de una dimensión
tikhonov_aprox = tikhonov_aprox.flatten()
problem = ImageReconstructionProblem(MODEL, PROBE, SIGNAL, n_var=n_var, tikhonov_aprox=None)

def binary_tournament(pop, P, **kwargs):
    # P define los torneos y los competidores
    n_tournaments, n_competitors = P.shape

    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")

    # Resultado que esta función debe devolver
    S = np.full(n_tournaments, -1, dtype=int)

    # Realizar todos los torneos
    for i in range(n_tournaments):
        a, b = P[i]

        # Si el primer individuo es mejor, elígelo
        if pop[a].F[0] < pop[b].F[0]:  
            S[i] = a
        else:
            S[i] = b

    return S

algorithm = NSGA2(
    pop_size=500,
    sampling=LHS(),
    crossover=CustomCrossover(prob=0.9),
    mutation=CustomMutation(prob=0.9, max_generations=1000),
    selection=TournamentSelection(func_comp=binary_tournament),
    eliminate_duplicates=True
)

res = minimize(problem,
               algorithm,
               seed=1,
               termination=('n_gen', 1000),
               verbose=True)

solutions = res.X
objective_values = res.F

min_index = np.argmin(objective_values[:, 0])

best_solution = solutions[min_index]
best_objective_values = objective_values[min_index]

if np.allclose(best_solution, tikhonov_aprox, atol=1e-10):
    print('The best solution is the Tikhonov approximation')

mu_est_nsga2 = problem.mo_estimation(best_solution.reshape(20, 1))
mu_est_tikh = problem.mo_estimation(tikhonov_aprox.reshape(20, 1))

print(f'NSGA-II Estimation (mu): {mu_est_nsga2}')
print(f'Tikhonov Estimation (mu): {mu_est_tikh}')

plt.figure(figsize=(10, 6))
plt.plot(mu_est_nsga2, label='NSGA-II Estimation', linestyle='-', color='b')
plt.plot(PROBE.mu, label='True mu', linestyle='--', color='g')
plt.plot(mu_est_tikh, label='Tikhonov Estimation', linestyle='-.', color='r')

plt.xlabel('Depth', fontsize=14)
plt.ylabel('mu', fontsize=14)
plt.title('Sin informar a NSGA-II (pymoo)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.savefig('img/nsga2_pro/mu_est_nsga2.png')
plt.close()

# Visualizar y guardar el frente de Pareto
scatter_plot = Scatter().add(res.F)
scatter_plot.save('pareto_front.png')