import csv
import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.sampling.lhs import LHS
from pymoo.indicators.hv import HV

from SSM.Generate_Linear_Model import generate_linear_model
from SSM.Generate_Measurements import generate_measurements
from SSM.Generate_SSM_Model import generate_ssm_model
from SSM.Set_Settings import set_settings
from SSM.Regularized_Estimation import regularized_estimation
from SSM.mu_from_d import mu_from_d

from SingleValue.SingleValueProblem import SingleValueReconstructionProblem
from MO.custom_crossover import CustomCrossover
from MO.custom_mutation import CustomMutation
from MO.binary_tournament import binary_tournament

from External_lib.Tikhonov import tikhonov


from utils.archive import update_archive, save_archive_pickle, limit_archive_size
from utils.plotting import plot_hypervolume, plot_pareto_front, plot_best_solution


def setup_problem(example=1, sigma=0.0):
    """Configura el problema de optimización"""
    model, PROBE, signal = set_settings(example=example)
    model = generate_ssm_model(model, PROBE)
    MODEL = generate_linear_model(model, signal, PROBE)
    SIGNAL = generate_measurements(signal, MODEL, sigma=sigma)
    d_lcurve, lambda_lcurve = regularized_estimation(MODEL, SIGNAL, dim=1)
    d_lcurve = d_lcurve.flatten()
    problem = SingleValueReconstructionProblem(MODEL, SIGNAL, l_curve_sol=None)
    return problem, PROBE, MODEL, SIGNAL, d_lcurve, lambda_lcurve


def record_hv_and_archive(algorithm, archive, hv_values, ref_point):
    """Callback para registrar el hipervolumen y actualizar el archivo externo"""
    if algorithm.pop is not None:
        archive = update_archive(algorithm.pop, archive)
        archive = limit_archive_size(archive, max_size=100)
        F = algorithm.pop.get("F")
        hv = HV(ref_point=ref_point).do(F)
        hv_values.append(hv)
    return archive


def run_optimization(problem, ref_point, archive_file, img_dir):
    """Ejecuta el algoritmo NSGA-II"""
    # Configurar el algoritmo
    algorithm = NSGA2(
        pop_size=500,
        sampling=LHS(),
        crossover=CustomCrossover(prob=1.0),
        mutation=PolynomialMutation(prob=0.83, eta=50),
        #selection=TournamentSelection(func_comp=binary_tournament),
        eliminate_duplicates=True
    )

    # Inicializar variables
    archive = []
    hv_values = []

    # Callback para actualizar el archivo y registrar el hipervolumen
    def callback(algorithm):
        nonlocal archive
        archive = record_hv_and_archive(algorithm, archive, hv_values, ref_point)

    # Ejecutar la optimización
    res = minimize(
        problem,
        algorithm,
        seed=1,
        termination=("n_gen", 500),
        verbose=True,
        callback=callback,
    )

    # Guardar el archivo externo
    save_archive_pickle(archive, archive_file)

    return res, archive, hv_values


def save_solutions_to_csv(l_curve_sol, lambda_lcurv, best_solution, best_index, objectives, solutions, output_file):
    """
    Save the L-curve solution and the best solution (based on weighted scoring) to a CSV file.

    Args:
        l_curve_sol (ndarray): Objective values of the solution given by the L-curve.
        best_solution (ndarray): Objective values of the best solution selected.
        best_index (int): Index of the best solution in the Pareto front.
        objectives (ndarray): All objective values from the Pareto front.
        output_file (str): Path to the CSV file for storing the results.
    """
    header = ["Solution Type", "lambda", "f1", "f2", "f3", "Index in Pareto Front"]

    # Collect data
    data = [
        ["L-Curve Solution"] + list(lambda_lcurv) + list(l_curve_sol) + ["N/A"],
        ["Best Solution"] + list(solutions[best_index]) + list(best_solution) + [best_index],
    ]

    # Add all Pareto front solutions for reference
    for i, obj in enumerate(objectives):
        data.append([f"Pareto Front Solution {i}"] + list(solutions[i]) + list(obj) + [i])

    # Write to CSV
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

def main():
    # Configuración
    ref_point = np.array([10, 10, 10])
    archive_file = "archive/single/external_archive_pro.pickle"
    img_dir = "img/nsga2_pro/single"

    # Configurar el problema
    problem, PROBE, MODEL, SIGNAL, d_lcurve, lambda_lcurve = setup_problem(example=1, sigma=0.05)

    # Ejecutar la optimización
    res, archive, hv_values = run_optimization(problem, ref_point, archive_file, img_dir)

    # Analizar resultados
    solutions = res.X
    objectives = res.F
    # Seleccionar la mejor solución basada en ponderaciones
    weights = [0.5, 0.3, 0.2]  # Example weights for objectives
    best_index, best_solution_values = problem.select_best_solution(objectives, weights=weights)
    best_solution = solutions[best_index]
    d_est_nsga2 = tikhonov(MODEL.H, SIGNAL.y, best_solution, dim=1)

    mu_est_nsga2 = mu_from_d(MODEL, d_est_nsga2)
    mu_est_l_curve = mu_from_d(MODEL, d_lcurve)
    
    l_curve_sol_val = problem.evaluate_l_curve_solution(lambda_lcurve)

    # Guardar las soluciones en un archivo CSV
    output_csv_file = "solutions_comparison.csv"
    save_solutions_to_csv(l_curve_sol_val, lambda_lcurve, best_solution_values, best_index, objectives, solutions, output_csv_file)


    # Graficar resultados
    plot_hypervolume(hv_values, img_dir)
    plot_pareto_front(res.F, archive=archive, img_dir=img_dir, lcurv_sol=l_curve_sol_val, best_sol=best_solution_values)
    plot_best_solution(img_dir, PROBE, mu_est_nsga2, mu_est_l_curve)
    


if __name__ == "__main__":
    main()
