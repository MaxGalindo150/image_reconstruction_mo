import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from External_lib.rmse import rmse


from SSM.Estimation import estimation
from SSM.Generate_Linear_Model import generate_linear_model
from SSM.Generate_Measurements import generate_measurements
from SSM.Generate_SSM_Model import generate_ssm_model
from SSM.Regularized_Estimation import regularized_estimation
from SSM.Set_Settings import set_settings

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.sampling.lhs import LHS

from MO.custom_crossover import CustomCrossover
from MO.custom_mutation import CustomMutation
from MO.MOP_Definition_Pymoo import ImageReconstructionProblem


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

# Configuración del modelo
n_var = 20
model, PROBE, signal = set_settings(example=1)
model = generate_ssm_model(model, PROBE)
MODEL = generate_linear_model(model, signal, PROBE)

# Valores de sigma (σ_w²) de 10^-10 a 10^-1
sigmas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
num_simulations = 10

# Para almacenar los resultados
avg_rmse_lsq = []
avg_rmse_nsga2 = []
avg_rmse_tikhonov = []
avg_rmse_tikhonov_nsga = []

# Función que corre una simulación
def run_simulation(i, sigma):
    # Generar mediciones con el ruido correspondiente
    SIGNAL = generate_measurements(signal, MODEL, sigma)
    
    # MO con NSGA-III
    problem = ImageReconstructionProblem(MODEL=MODEL, PROBE=PROBE, SIGNAL=SIGNAL, n_var=n_var)
    # Crear las direcciones de referencia para la optimización
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
    # Crear el objeto del algoritmo
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
                termination=('n_gen', 1000))
    solutions = res.X
    objective_values = res.F
    # Encontrar el índice de la solución con el menor valor en el objetivo 1
    min_index = np.argmin(objective_values[:, 0])
    # Obtener la solución correspondiente a ese índice
    d_est_nsga = solutions[min_index]
    best_objective_values = objective_values[min_index]
    mu_est_nsga = problem.mo_estimation(d_est_nsga.reshape(20,1))
    
    # LSQ
    ESTIMATION_RESULTS = estimation(MODEL, SIGNAL)
    mu_est_lsq = ESTIMATION_RESULTS.mu

    # Tikhonov
    d_est_tikh = regularized_estimation(MODEL, SIGNAL, dim=1).reshape(n_var, 1)
    mu_est_tikh = problem.mo_estimation(d_est_tikh.reshape(20,1))

    #Thikonov and NSGA
    problem = ImageReconstructionProblem(MODEL=MODEL, PROBE=PROBE, SIGNAL=SIGNAL, n_var=n_var, tikhonov_aprox=d_est_tikh)
    # Crear las direcciones de referencia para la optimización
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
    # Crear el objeto del algoritmo
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
                termination=('n_gen', 1000))
    solutions = res.X
    objective_values = res.F
    # Encontrar el índice de la solución con el menor valor en el objetivo 1
    min_index = np.argmin(objective_values[:, 0])
    # Obtener la solución correspondiente a ese índice
    d_est_nsga = solutions[min_index]
    best_objective_values = objective_values[min_index]
    mu_est_tikh_nsga = problem.mo_estimation(d_est_nsga.reshape(20,1))
    

    # Calcular RMSE para cada método
    rmse_lsq = rmse(PROBE.mu, mu_est_lsq)
    rmse_nsga2 = rmse(PROBE.mu, mu_est_nsga)
    rmse_tikhonov = rmse(PROBE.mu, mu_est_tikh)
    rmse_tikhonov_nsga = rmse(PROBE.mu, mu_est_tikh_nsga)

    return (rmse_lsq, rmse_nsga2, rmse_tikhonov, rmse_tikhonov_nsga)


# Ejecutar simulaciones en paralelo con barra de progreso
with open("results.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sigma", "lse", "nsga2", "tikhonov", "tikhonov_nsga"])
    
    for sigma in sigmas:
        rmse_lsq_list = []
        rmse_nsga2_list = []
        rmse_tikhonov_list = []
        rmse_tikhonov_nsga_list = []
        
        # Crear el pool de procesos
        with ProcessPoolExecutor() as executor:
            # Usar tqdm para visualizar el progreso
            futures = [executor.submit(run_simulation, i, sigma) for i in range(num_simulations)]
            
            # tqdm as_completed para mostrar progreso de las simulaciones completadas
            for future in tqdm(as_completed(futures), total=num_simulations, desc=f"Simulaciones para sigma={sigma}"):
                rmse_lsq, rmse_nsga2, rmse_tikhonov, rmse_tikhonov_nsga = future.result()
                rmse_lsq_list.append(rmse_lsq)
                rmse_nsga2_list.append(rmse_nsga2)
                rmse_tikhonov_list.append(rmse_tikhonov)
                rmse_tikhonov_nsga_list.append(rmse_tikhonov_nsga)

        # Promediar los RMSE de las simulaciones
        avg_rmse_lsq.append(np.mean(rmse_lsq_list))
        avg_rmse_nsga2.append(np.mean(rmse_nsga2_list))
        avg_rmse_tikhonov.append(np.mean(rmse_tikhonov_list))
        avg_rmse_tikhonov_nsga.append(np.mean(rmse_tikhonov_nsga_list))

        writer.writerow([sigma, avg_rmse_lsq[-1], avg_rmse_nsga2[-1], avg_rmse_tikhonov[-1], avg_rmse_tikhonov_nsga[-1]])    

# Graficar los resultados
plt.figure()
plt.plot(sigmas, avg_rmse_lsq, label='LSQ', marker='o')
plt.plot(sigmas, avg_rmse_nsga2, label='NSGA-II', marker='x')
plt.plot(sigmas, avg_rmse_tikhonov, label='Tikhonov', marker='s')
plt.plot(sigmas, avg_rmse_tikhonov_nsga, label='Tikhonov + NSGA-III', marker='d')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\sigma_w^2$')
plt.ylabel('Average RMSE')
plt.legend()
plt.grid(True)

# Guardar la gráfica
plt.savefig("img/impact_of_noise_on_rmse_2.png")
