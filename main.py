import numpy as np
import matplotlib.pyplot as plt
from External_lib.rmse import rmse
import csv
from MO.MOP_Definition_Pymoo import ImageReconstructionProblem as ImageReconstructionProblemPymoo
from MO.MOP_Definition import ImageReconstructionProblem
from MO.NSGA_II import NSGA2
from SSM.Estimation import estimation
from SSM.Generate_Linear_Model import generate_linear_model
from SSM.Generate_Measurements import generate_measurements
from SSM.Generate_SSM_Model import generate_ssm_model
from SSM.Regularized_Estimation import regularized_estimation
from SSM.Set_Settings import set_settings

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

model, PROBE, signal = set_settings(example=1)

# Generate SSM model
model = generate_ssm_model(model, PROBE)

# Generate linear model
MODEL = generate_linear_model(model, signal, PROBE)

# Generate measurements
SIGNAL = generate_measurements(signal, MODEL)

# Estimation
ESTIMATION_RESULTS = estimation(MODEL, SIGNAL)

# =================================================== MO ===================================================
# Problem
# PROBLEM = ImageReconstructionProblem(MODEL, PROBE, SIGNAL, n_var=20, b=None)

# # NSGA-II
# nsga2 = NSGA2(generations=460, population_size=100, mutation_rate=0.8, problem=PROBLEM, objective1_threshold=0.01)
# x = [individual.values[0] for individual in nsga2.P_t]
# y = [individual.values[1] for individual in nsga2.P_t]

# best_individual_nsga2 = min(nsga2.P_t, key=lambda p: p.values[0])
# best_point_nsga2 = best_individual_nsga2.point

# mu_est_nsga2 = PROBLEM.mo_estimation(best_point_nsga2.reshape(20, 1))

# plt.figure()
# plt.scatter(x, y, c='red')
# plt.xlabel('f1')
# plt.ylabel('f2')
# plt.savefig('img/pareto_front.png')

# =================================================== Simulations ===================================================
sigmas = [0.0, 1e-4, 1e-3, 1e-3, 1e-1, 1.0]
n_var = 20

with open("profiles_sigma.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sigma", "lse", "nsga2", "tikhonov"])

    for sigma in sigmas:
        # Generar mediciones con el ruido correspondiente
        SIGNAL = generate_measurements(signal, MODEL, sigma)
        
        # MO con NSGA-III
        PROBLEM = ImageReconstructionProblem(MODEL, PROBE, SIGNAL, n_var=20, b=None)

        # # NSGA-II
        nsga2 = NSGA2(generations=460, population_size=100, mutation_rate=0.8, problem=PROBLEM, objective1_threshold=0.01)
        x = [individual.values[0] for individual in nsga2.P_t]
        y = [individual.values[1] for individual in nsga2.P_t]

        best_individual_nsga2 = min(nsga2.P_t, key=lambda p: p.values[0])
        best_point_nsga2 = best_individual_nsga2.point

        mu_est_nsga = PROBLEM.mo_estimation(best_point_nsga2.reshape(20, 1))
        
        # LSQ
        ESTIMATION_RESULTS = estimation(MODEL, SIGNAL)
        mu_est_lsq = ESTIMATION_RESULTS.mu

        # Tikhonov
        d_est_tikh = regularized_estimation(MODEL, SIGNAL, dim=1).reshape(n_var, 1)
        mu_est_tikh = PROBLEM.mo_estimation(d_est_tikh.reshape(20,1))

        #Thikonov and NSGA
        problem = ImageReconstructionProblemPymoo(MODEL=MODEL, PROBE=PROBE, SIGNAL=SIGNAL, n_var=n_var, tikhonov_aprox=d_est_tikh)
        # Crear las direcciones de referencia para la optimización
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
        # Crear el objeto del algoritmo
        algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)
        # Ejecutar la optimización
        res = minimize(problem,
                    algorithm,
                    seed=1,
                    termination=('n_gen', 2500))
        solutions = res.X
        objective_values = res.F
        # Encontrar el índice de la solución con el menor valor en el objetivo 1
        min_index = np.argmin(objective_values[:, 0])
        # Obtener la solución correspondiente a ese índice
        d_est_nsga = solutions[min_index]
        best_objective_values = objective_values[min_index]
        mu_est_tikh_nsga = problem.mo_estimation(d_est_nsga.reshape(20,1))

        writer.writerow([sigma, rmse(PROBE.mu, mu_est_lsq), rmse(PROBE.mu, mu_est_nsga), rmse(PROBE.mu, mu_est_tikh), rmse(PROBE.mu, mu_est_tikh_nsga)])

        # Generar y guardar figuras para cada sigma
        plt.figure()
        plt.plot(PROBE.mu, "b-", linewidth=4, label="True µ")
        plt.plot(mu_est_nsga, 'g-', label='Estimated µ by NSGA-II')
        # plt.plot(mu_est_lse, 'r--', label='Estimated µ by LSE')
        plt.plot(mu_est_tikh, "k-", label="Estimated µ by Tikhonov")
        plt.plot(mu_est_tikh_nsga, "m-", label="Estimated µ by Tikhonov and NSGA-II")
        plt.legend()
        plt.title(f"Estimation Results for sigma={sigma}")
        plt.savefig(f"img/1D/true_and_estimated_mu_sigma_{sigma}.png")
        plt.close()

        plt.figure()
        plt.plot(SIGNAL.i, "r--", label="i")
        plt.plot(SIGNAL.y, "k-", label="y")
        plt.legend()
        plt.grid(True)
        plt.title(f"Laser Signal Measurement for sigma={sigma}")
        plt.savefig(f"img/1D/laser_signal_measurement_signal_sigma_{sigma}.png")
        plt.close()
