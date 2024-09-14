# import matplotlib.pyplot as plt
# from External_lib.rmse import rmse
# import csv
# from MO.MOP_Definition import ImageReconstructionProblem
# from MO.NSGA_II import NSGA2
# from SSM.Estimation import estimation
# from SSM.Generate_Linear_Model import generate_linear_model
# from SSM.Generate_Measurements import generate_measurements
# from SSM.Generate_SSM_Model import generate_ssm_model
# from SSM.Regularized_Estimation import regularized_estimation
# from SSM.Set_Settings import set_settings
#
# model, PROBE, signal = set_settings(example=1)
#
# # Generate SSM model
# model = generate_ssm_model(model, PROBE)
#
# # Generate linear model
# MODEL = generate_linear_model(model, signal, PROBE)
#
# # Generate measurements
# SIGNAL = generate_measurements(signal, MODEL)
#
# # Estimation
# ESTIMATION_RESULTS = estimation(MODEL, SIGNAL)
#
# # =================================================== MO ===================================================
# #Problem
# # PROBLEM = ImageReconstructionProblem(MODEL, PROBE, SIGNAL)
# #
# # # NSGA-II
# # nsga2 = NSGA2(generations=500, population_size=100, mutaition_rate=0.8, problem=PROBLEM)
# # x = [individual.values[0] for individual in nsga2.P_t]
# # y = [individual.values[1] for individual in nsga2.P_t]
# #
# # best_individual_nsga2 = min(nsga2.P_t, key=lambda p: p.values[0])
# # best_point_nsga2 = best_individual_nsga2.point
# #
# # mu_est_nsga2 = PROBLEM.mo_estimation(best_point_nsga2.reshape(20, 1))
# #
# #
# # plt.figure()
# # plt.scatter(x, y, c='red')
# # plt.xlabel('f1')
# # plt.ylabel('f2')
# # #plt.title('Pareto Front')
# # plt.savefig('pareto_front.png')
# #
#
#
#
# # =================================================== Simulations ===================================================
# sigmas = [0.0, 1e-4, 1e-3, 1e-3, 1e-1, 1.0]
#
# d_estimations = []
# mu_estimations = []
#
#
# with open("results.csv", mode='w', newline='') as file:
#
#     writer = csv.writer(file)
#     writer.writerow(["sigma", "lsq", "nsga2", "tikhonov"])
#
#     for sigma in sigmas:
#
#         SIGNAL = generate_measurements(signal, MODEL, sigma)
#
#
#         # MO
#         PROBLEM = ImageReconstructionProblem(MODEL, PROBE, SIGNAL)
#         nsga2 = NSGA2(generations=500, population_size=100, mutaition_rate=0.8, problem=PROBLEM)
#         x = [individual.values[0] for individual in nsga2.P_t]
#         y = [individual.values[1] for individual in nsga2.P_t]
#         best_individual_nsga2 = min(nsga2.P_t, key=lambda p: p.values[0])
#         d_est_nsga2 = best_individual_nsga2.point.reshape(20, 1)
#
#         mu_est_nsga2 = PROBLEM.mo_estimation(d_est_nsga2)
#
#
#         # LSQ
#         ESTIMATION_RESULTS = estimation(MODEL, SIGNAL)
#         d_est_lsq = ESTIMATION_RESULTS.d
#         mu_est_lsq = ESTIMATION_RESULTS.mu
#
#         # Tikhonov
#         mu_est_tikh = regularized_estimation(MODEL, SIGNAL)
#
#         writer.writerow([sigma, rmse(PROBE.mu, mu_est_lsq), rmse(PROBE.mu, mu_est_nsga2), rmse(PROBE.mu, mu_est_tikh)])
#
#
#
#
# # Regularized estimation
# mu_est_tikh = regularized_estimation(MODEL, SIGNAL)
# #
# # # Primera figura
# plt.figure()
# plt.plot(PROBE.mu, "b-", linewidth=4, label="True µ")
# # plt.plot(mu_est_nsga2, 'g-', label='Estimated µ by NSGA-II')
# # # plt.plot(ESTIMATION_RESULTS.mu, 'r--', label='Estimated µ')
# print("true mu", PROBE.mu)
# # print("true d: ", PROBE.d)
# plt.plot(mu_est_tikh, "k-", label="Estimated µ by Tikhonov")
# plt.legend()
# plt.savefig("true_and_estimated_mu.png")
# plt.close()  # Cierra la primera figura
#
# # Segunda figura
# plt.figure()
# plt.plot(SIGNAL.i, "r--", label="i")
# plt.plot(SIGNAL.y, "k-", label="y")
# plt.legend()
# plt.grid(True)
# plt.savefig("laser_signal_measurement_signal.png")
# plt.close()  # Cierra la segunda figura
#

import numpy as np
import matplotlib.pyplot as plt
import csv
from External_lib.rmse import rmse
from MO.MOP_Definition import ImageReconstructionProblem
from MO.NSGA_II import NSGA2
from SSM.Estimation import estimation
from SSM.Generate_Linear_Model import generate_linear_model
from SSM.Generate_Measurements import generate_measurements
from SSM.Generate_SSM_Model import generate_ssm_model
from SSM.Regularized_Estimation import regularized_estimation
from SSM.Set_Settings import set_settings

# Configuración del modelo
model, PROBE, signal = set_settings(example=1)
model = generate_ssm_model(model, PROBE)
MODEL = generate_linear_model(model, signal, PROBE)

# Valores de sigma (σ_w²) de 10^-10 a 10^-1
# sigmas = np.logspace(-10, -1, 10)
sigmas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
# Número de simulaciones por cada valor de sigma
num_simulations = 100

# Para almacenar los resultados
avg_rmse_lsq = []
avg_rmse_nsga2 = []
avg_rmse_tikhonov = []
count = 0

with open("results.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sigma", "lsq", "nsga2", "tikhonov"])
    for sigma in sigmas:
        rmse_lsq_list = []
        rmse_nsga2_list = []
        rmse_tikhonov_list = []

        for i in range(num_simulations):
       
            if (i + 1) % 10 == 0:
                print(f"Simulación {count + 1} de {num_simulations * len(sigmas)}")
                count += 1
            # print(f"Simulación {i + 1} de {num_simulations} para sigma = {sigma}")
            # Generar mediciones con el ruido correspondiente
            SIGNAL = generate_measurements(signal, MODEL, sigma)
        
            # MO con NSGA-II
            PROBLEM = ImageReconstructionProblem(MODEL, PROBE, SIGNAL)
            nsga2 = NSGA2(generations=500, population_size=100, mutaition_rate=0.8, problem=PROBLEM)
            best_individual_nsga2 = min(nsga2.P_t, key=lambda p: p.values[0])
            d_est_nsga2 = best_individual_nsga2.point.reshape(20, 1)
            mu_est_nsga2 = PROBLEM.mo_estimation(d_est_nsga2)

            # LSQ
            ESTIMATION_RESULTS = estimation(MODEL, SIGNAL)
            mu_est_lsq = ESTIMATION_RESULTS.mu
        
            # Tikhonov
            mu_est_tikh = regularized_estimation(MODEL, SIGNAL)
        
            # Calcular RMSE para cada método
            rmse_lsq_list.append(rmse(PROBE.mu, mu_est_lsq))
            rmse_nsga2_list.append(rmse(PROBE.mu, mu_est_nsga2))
            rmse_tikhonov_list.append(rmse(PROBE.mu, mu_est_tikh))
    
        # Promediar los RMSE de las 100 simulaciones
        avg_rmse_lsq.append(np.mean(rmse_lsq_list))
        avg_rmse_nsga2.append(np.mean(rmse_nsga2_list))
        avg_rmse_tikhonov.append(np.mean(rmse_tikhonov_list))

        writer.writerow([sigma, avg_rmse_lsq[-1], avg_rmse_nsga2[-1], avg_rmse_tikhonov[-1]])






# print("avg_rmse_lsq: ", avg_rmse_nsga2)
# Graficar los resultados
plt.figure()
plt.plot(sigmas, avg_rmse_lsq, label='LSQ', marker='o')
plt.plot(sigmas, avg_rmse_nsga2, label='NSGA-II', marker='x')
plt.plot(sigmas, avg_rmse_tikhonov, label='Tikhonov', marker='s')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\sigma_w^2$')
plt.ylabel('Average RMSE')
plt.legend()
plt.grid(True)

# Guardar la gráfica
plt.savefig("impact_of_noise_on_rmse.png")
# plt.show()


