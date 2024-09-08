import matplotlib.pyplot as plt

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
#Problem
PROBLEM = ImageReconstructionProblem(MODEL, PROBE, SIGNAL)

# NSGA-II
nsga2 = NSGA2(generations=500, population_size=100, mutaition_rate=0.8, problem=PROBLEM)
x = [individual.values[0] for individual in nsga2.P_t]
y = [individual.values[1] for individual in nsga2.P_t]

best_individual_nsga2 = min(nsga2.P_t, key=lambda p: p.values[0])
best_point_nsga2 = best_individual_nsga2.point

mu_est_nsga2 = PROBLEM.mo_estimation(best_point_nsga2.reshape(20, 1))


plt.figure()
plt.scatter(x, y, c='red')
plt.xlabel('f1')
plt.ylabel('f2')
#plt.title('Pareto Front')
plt.savefig('pareto_front.png')

# Regularized estimation
mu_est_tikh = regularized_estimation(MODEL, SIGNAL)
#
# # Primera figura
plt.figure()
plt.plot(PROBE.mu, "b-", linewidth=4, label="True µ")
plt.plot(mu_est_nsga2, 'g-', label='Estimated µ by NSGA-II')
# # plt.plot(ESTIMATION_RESULTS.mu, 'r--', label='Estimated µ')
print("true mu", PROBE.mu)
# print("true d: ", PROBE.d)
plt.plot(mu_est_tikh, "k-", label="Estimated µ by Tikhonov")
plt.legend()
plt.savefig("true_and_estimated_mu.png")
plt.close()  # Cierra la primera figura

# Segunda figura
plt.figure()
plt.plot(SIGNAL.i, "r--", label="i")
plt.plot(SIGNAL.y, "k-", label="y")
plt.legend()
plt.grid(True)
plt.savefig("laser_signal_measurement_signal.png")
plt.close()  # Cierra la segunda figura
