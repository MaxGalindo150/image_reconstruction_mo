import matplotlib.pyplot as plt
from External_lib.rmse import rmse
import csv
from MO.MOP_Definition import ImageReconstructionProblem
from MO.NSGA_II import NSGA2
from SSM.Estimation import estimation
from SSM.Generate_Linear_Model import generate_linear_model
from SSM.Generate_Measurements import generate_measurements
from SSM.Generate_SSM_Model import generate_ssm_model
from SSM.Regularized_Estimation import regularized_estimation
from SSM.Set_Settings import set_settings

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
PROBLEM = ImageReconstructionProblem(MODEL, PROBE, SIGNAL)

# NSGA-II
nsga2 = NSGA2(generations=460, population_size=100, mutaition_rate=0.8, problem=PROBLEM, objective1_threshold=0.01)
x = [individual.values[0] for individual in nsga2.P_t]
y = [individual.values[1] for individual in nsga2.P_t]

best_individual_nsga2 = min(nsga2.P_t, key=lambda p: p.values[0])
best_point_nsga2 = best_individual_nsga2.point

mu_est_nsga2 = PROBLEM.mo_estimation(best_point_nsga2.reshape(20, 1))

plt.figure()
plt.scatter(x, y, c='red')
plt.xlabel('f1')
plt.ylabel('f2')
plt.savefig('img/pareto_front.png')

# =================================================== Simulations ===================================================
sigmas = [0.0, 1e-4, 1e-3, 1e-3, 1e-1, 1.0]

with open("results2.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sigma", "lse", "nsga2", "tikhonov"])

    for sigma in sigmas:
        SIGNAL = generate_measurements(signal, MODEL, sigma)

        # MO
        PROBLEM = ImageReconstructionProblem(MODEL, PROBE, SIGNAL)
        nsga2 = NSGA2(generations=500, population_size=100, mutaition_rate=0.8, problem=PROBLEM, objective1_threshold=0.01)
        x = [individual.values[0] for individual in nsga2.P_t]
        y = [individual.values[1] for individual in nsga2.P_t]
        best_individual_nsga2 = min(nsga2.P_t, key=lambda p: p.values[0])
        d_est_nsga2 = best_individual_nsga2.point.reshape(20, 1)

        mu_est_nsga2 = PROBLEM.mo_estimation(d_est_nsga2)

        # LSE
        ESTIMATION_RESULTS = estimation(MODEL, SIGNAL)
        # d_est_lse = ESTIMATION_RESULTS.d
        mu_est_lse = ESTIMATION_RESULTS.mu

        # Tikhonov
        mu_est_tikh = regularized_estimation(MODEL, SIGNAL)

        writer.writerow([sigma, rmse(PROBE.mu, mu_est_lse), rmse(PROBE.mu, mu_est_nsga2), rmse(PROBE.mu, mu_est_tikh)])

        # Generar y guardar figuras para cada sigma
        plt.figure()
        plt.plot(PROBE.mu, "b-", linewidth=4, label="True µ")
        plt.plot(mu_est_nsga2, 'g-', label='Estimated µ by NSGA-II')
        # plt.plot(mu_est_lse, 'r--', label='Estimated µ by LSE')
        plt.plot(mu_est_tikh, "k-", label="Estimated µ by Tikhonov")
        plt.legend()
        plt.title(f"Estimation Results for sigma={sigma}")
        plt.savefig(f"img/true_and_estimated_mu_sigma_{sigma}.png")
        plt.close()

        plt.figure()
        plt.plot(SIGNAL.i, "r--", label="i")
        plt.plot(SIGNAL.y, "k-", label="y")
        plt.legend()
        plt.grid(True)
        plt.title(f"Laser Signal Measurement for sigma={sigma}")
        plt.savefig(f"img/laser_signal_measurement_signal_sigma_{sigma}.png")
        plt.close()
