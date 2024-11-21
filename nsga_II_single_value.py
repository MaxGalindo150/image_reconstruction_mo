import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.sampling.lhs import LHS

from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.ux import UniformCrossover


from SSM.Generate_Linear_Model import generate_linear_model
from SSM.Generate_Measurements import generate_measurements
from SSM.Generate_SSM_Model import generate_ssm_model
from SSM.Set_Settings import set_settings
from SSM.Regularized_Estimation import regularized_estimation
from SSM.mu_from_d import mu_from_d

from External_lib.Tikhonov import tikhonov

from MO.custom_crossover import CustomCrossover
from MO.custom_mutation import CustomMutation

from SingleValue.SingleValueProblem import SingleValueReconstructionProblem

# Generar datos de ejemplo
model, PROBE, signal = set_settings(example=1)
model = generate_ssm_model(model, PROBE)
MODEL = generate_linear_model(model, signal, PROBE)
SIGNAL = generate_measurements(signal, MODEL, sigma=0.05)
d_est_l_curve, l_curv_sol = regularized_estimation(MODEL, SIGNAL, dim=1)
d_est_l_curve = d_est_l_curve.flatten()

# Definir el problema
problem = SingleValueReconstructionProblem(MODEL, SIGNAL, l_curve_sol=None)

algorithm = NSGA2(
    pop_size=500,
    sampling=LHS(),
    crossover=CustomCrossover(prob=1.0),
    mutation=PolynomialMutation(prob=0.83, eta=50),
    eliminate_duplicates=True
)

res = minimize(problem,
               algorithm,
               seed=1,
               termination=('n_gen', 1000),
               verbose=True)


# Obtener los resultados
solutions = res.X  # Valores de lambda en el frente de Pareto
objectives = res.F  # Valores de los objetivos correspondientes

l_curv_values = problem.evaluate_l_curve_solution(l_curv_sol)

min_index = np.argmin(objectives[:, 0])

lambda_first_obj = solutions[min_index]
best_objective_values = objectives[min_index]


d_est_nsga2 = tikhonov(MODEL.H, SIGNAL.y, lambda_first_obj, dim=1)

print(f"lambda_ngsa_ii: {lambda_first_obj}")
print(f"lambda_l_curve: {l_curv_sol}")

mu_est_nsga2 = mu_from_d(MODEL, d_est_nsga2)
mu_est_l_curve = mu_from_d(MODEL, d_est_l_curve)

plt.figure(figsize=(10, 6))
plt.plot(mu_est_nsga2, label='NSGA-II Estimation', linestyle='-', color='b')
plt.plot(PROBE.mu, label='True mu', linestyle='--', color='g')
plt.plot(mu_est_l_curve, label='L-curve Estimation', linestyle='-.', color='r')

plt.xlabel('Depth', fontsize=14)
plt.ylabel('mu', fontsize=14)
plt.title('Sin informar a NSGA-II (pymoo)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.savefig('img/nsga2_pro/mu_est_nsga2_single.png')
plt.close()

if np.allclose(mu_est_nsga2, mu_est_l_curve, atol=1e-10):
    print('The best solution is the Tikhonov approximation')


# Graficar el frente de Pareto
l_curv_values_norm = problem.evaluate_l_curve_solution(l_curv_sol)

# Graficar el frente de Pareto
scatter = Scatter(title="Pareto Front").add(objectives, label="Pareto Front")

# Agregar la solución de la L-Curve al gráfico
scatter.add(l_curv_values_norm, label="L-Curve Solution", color="orange")

# Mostrar y guardar el gráfico
scatter.save("pareto_front_single_value.png")
scatter.show()



results_df = pd.DataFrame({
    "lambda": solutions.flatten(),  # Valores de lambda
    "f1": objectives[:, 0],         # Primer objetivo (residual)
    "f2": objectives[:, 1],         # Segundo objetivo (regularización)
    "f3": objectives[:, 2]          # Tercer objetivo (positividad)
})

# Guardar en un archivo CSV
results_df.to_csv("results_pareto_single_value.csv", index=False)
print("Resultados guardados en results_pareto_single_value.csv")
