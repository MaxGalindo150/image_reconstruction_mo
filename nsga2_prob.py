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
PROBLEM = ImageReconstructionProblem(MODEL, PROBE, SIGNAL, n_var=20, b=None)

# NSGA-II
nsga2 = NSGA2(generations=500, population_size=100, mutation_rate=0.8, problem=PROBLEM, objective1_threshold=0.01)
x = [individual.values[0] for individual in nsga2.P_t]
y = [individual.values[1] for individual in nsga2.P_t]

best_individual_nsga2 = min(nsga2.P_t, key=lambda p: p.values[0])
best_point_nsga2 = best_individual_nsga2.point

mu_est_nsga2 = PROBLEM.mo_estimation(best_point_nsga2.reshape(20, 1))

plt.figure()
plt.scatter(x, y, c='red')
plt.xlabel('f1')
plt.ylabel('f2')
plt.savefig('img/nsga2_pro/pareto_front.png')
plt.close()

plt.figure()
plt.plot(mu_est_nsga2)
plt.plot(ESTIMATION_RESULTS.mu)
plt.xlabel('Depth')
plt.ylabel('mu')
plt.savefig('img/nsga2_pro/mu_est_nsga2.png')
plt.close()

