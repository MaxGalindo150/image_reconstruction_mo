import numpy as np
import matplotlib.pyplot as plt
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from External_lib.rmse import rmse
from External_lib.Tikhonov import tikhonov

from SSM.mu_from_d import mu_from_d
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
from pymoo.operators.mutation.pm import PolynomialMutation

from MO.custom_crossover import CustomCrossover
from MO.custom_mutation import CustomMutation
from SingleValue.SingleValueProblem import SingleValueReconstructionProblem


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
    problem = SingleValueReconstructionProblem(MODEL=MODEL, SIGNAL=SIGNAL)
    
    algorithm = NSGA2(
        pop_size=500,
        sampling=LHS(),
        crossover=CustomCrossover(prob=1.0),
        mutation=PolynomialMutation(prob=0.83, eta=50),
        #selection=TournamentSelection(func_comp=binary_tournament),
        eliminate_duplicates=True
    )
    
    res = minimize(problem,
                algorithm,
                seed=1,
                termination=('n_gen', 1000))
    
    solutions = res.X
    objectives = res.F
    

    min_index = np.argmin(objectives[:, 0])

    lambda_first_obj = solutions[min_index]
    best_objective_values = objectives[min_index]
    
    # L-Curve
    d_est_l_curve, l_curve_sol = regularized_estimation(MODEL, SIGNAL, dim=1)
    # NSGA-II
    d_est_nsga2 = tikhonov(MODEL.H, SIGNAL.y, lambda_first_obj, dim=1)
    
    # LSQ
    ESTIMATION_RESULTS = estimation(MODEL, SIGNAL)
    mu_est_lsq = ESTIMATION_RESULTS.mu

    mu_est_nsga2 = problem.mo_estimation(d_est_nsga2)
    mu_est_l_curve = problem.mo_estimation(d_est_l_curve)
    
    #Thikonov and NSGA
    problem = SingleValueReconstructionProblem(MODEL=MODEL, SIGNAL=SIGNAL, l_curve_sol=l_curve_sol)
    # Crear las direcciones de referencia para la optimización
    algorithm = NSGA2(
        pop_size=500,
        sampling=LHS(),
        crossover=CustomCrossover(prob=1.0),
        mutation=PolynomialMutation(prob=0.83, eta=50),
        #selection=TournamentSelection(func_comp=binary_tournament),
        eliminate_duplicates=True
    ) 
    res = minimize(problem,
                algorithm,
                seed=1,
                termination=('n_gen', 1000))
    solutions = res.X
    objectives = res.F
    # Encontrar el índice de la solución con el menor valor en el objetivo 1
    min_index = np.argmin(objectives[:, 0])
    # Obtener la solución correspondiente a ese índice
    lambda_first_obj = solutions[min_index]
    best_objective_values = objectives[min_index]
    d_est_nsga_l_curve = tikhonov(MODEL.H, SIGNAL.y, lambda_first_obj, dim=1)
    mu_est_nsga_l_curve = problem.mo_estimation(d_est_nsga_l_curve) 

    # Calcular RMSE para cada método
    rmse_lsq = rmse(PROBE.mu, mu_est_lsq)
    rmse_nsga2 = rmse(PROBE.mu, mu_est_nsga2)
    rmse_l_curve = rmse(PROBE.mu, mu_est_l_curve)
    rmse_nsga2_l_curve = rmse(PROBE.mu, mu_est_nsga_l_curve)

    return (rmse_lsq, rmse_nsga2, rmse_l_curve, rmse_nsga2_l_curve)


# Ejecutar simulaciones en paralelo con barra de progreso
with open("results.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sigma", "lse", "nsga2", "l_curve", "nsga2_l_curve"])
    
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
plt.savefig("img/impact_of_noise_on_rmse_single.png")
