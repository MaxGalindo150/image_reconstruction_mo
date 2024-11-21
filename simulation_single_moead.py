import numpy as np
import matplotlib.pyplot as plt
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from External_lib.rmse import rmse
from External_lib.Tikhonov import tikhonov

from SSM.Estimation import estimation
from SSM.Generate_Linear_Model import generate_linear_model
from SSM.Generate_Measurements import generate_measurements
from SSM.Generate_SSM_Model import generate_ssm_model
from SSM.Regularized_Estimation import regularized_estimation
from SSM.Set_Settings import set_settings

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.algorithms.moo.moead import MOEAD


from MO.custom_crossover import CustomCrossover
from MO.custom_mutation import CustomMutation
from SingleValue.SingleValueProblem import SingleValueReconstructionProblem


from utils.plotting import plot_rmse_vs_noise


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

# Función que corre una simulación
def run_simulation(i, sigma):
    # Generar mediciones con el ruido correspondiente
    SIGNAL = generate_measurements(signal, MODEL, sigma)
    
    # MO con MOEA/D
    problem = SingleValueReconstructionProblem(MODEL=MODEL, SIGNAL=SIGNAL)
    
    # Generar direcciones de referencia con un número válido de puntos
    ref_dirs = get_reference_directions("uniform", 3, n_points=105)

    # Crear instancia del algoritmo MOEA/D
    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=20,
        #decomposition=PenaltyBoundaryIntersection(),  
        prob_neighbor_mating=0.9,
        #sampling=LHS(),
        mutation=PolynomialMutation(prob=83, eta=50),
    )

    
    res = minimize(problem,
                   algorithm,
                   seed=1,
                   termination=('n_gen', 500))
    
    solutions = res.X
    objectives = res.F
    
    weights = [0.5, 0.25, 0.25]
    min_index, best_sol_values = problem.select_best_solution(objectives, weights=weights)
    
    # Seleccionar la mejor solución basada en el primer objetivo
    #min_index = np.argmin(objectives[:, 0])
    lambda_first_obj = solutions[min_index]
    
    # L-Curve
    d_est_l_curve, l_curve_sol = regularized_estimation(MODEL, SIGNAL, dim=1)
    # MOEA/D
    d_est_moead = tikhonov(MODEL.H, SIGNAL.y, lambda_first_obj, dim=1)
    
    mu_est_moead = problem.mo_estimation(d_est_moead)
    mu_est_l_curve = problem.mo_estimation(d_est_l_curve)
    
    # LSQ
    ESTIMATION_RESULTS = estimation(MODEL, SIGNAL)
    mu_est_lsq = ESTIMATION_RESULTS.mu

    # Calcular RMSE para cada método
    rmse_lsq = rmse(PROBE.mu, mu_est_lsq)
    rmse_moead = rmse(PROBE.mu, mu_est_moead)
    rmse_l_curve = rmse(PROBE.mu, mu_est_l_curve)
    
    return (rmse_lsq, rmse_moead, rmse_l_curve)


# Ejecutar simulaciones en paralelo con barra de progreso
with open("results_moead.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sigma", "lse", "moead", "l_curve"])
    
    for sigma in sigmas:
        rmse_lsq_list = []
        rmse_moead_list = []
        rmse_tikhonov_list = []
        
        # Crear el pool de procesos
        with ProcessPoolExecutor() as executor:
            # Usar tqdm para visualizar el progreso
            futures = [executor.submit(run_simulation, i, sigma) for i in range(num_simulations)]
            
            # tqdm as_completed para mostrar progreso de las simulaciones completadas
            for future in tqdm(as_completed(futures), total=num_simulations, desc=f"Simulaciones para sigma={sigma}"):
                rmse_lsq, rmse_moead, rmse_tikhonov = future.result()
                rmse_lsq_list.append(rmse_lsq)
                rmse_moead_list.append(rmse_moead)
                rmse_tikhonov_list.append(rmse_tikhonov)

        # Promediar los RMSE de las simulaciones
        avg_rmse_lsq.append(np.mean(rmse_lsq_list))
        avg_rmse_nsga2.append(np.mean(rmse_moead_list))
        avg_rmse_tikhonov.append(np.mean(rmse_tikhonov_list))
        
        writer.writerow([sigma, avg_rmse_lsq[-1], avg_rmse_nsga2[-1], avg_rmse_tikhonov[-1]])    

# Graficar los resultados
plot_rmse_vs_noise(sigmas, avg_rmse_lsq, avg_rmse_nsga2, avg_rmse_tikhonov, "MOEA/D", "img/nsga2_pro/single/moead")