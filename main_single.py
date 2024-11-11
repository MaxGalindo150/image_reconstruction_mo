import json
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


def setup_problem():
    """Configura el problema de optimización"""
    model, PROBE, signal = set_settings(example=1)
    model = generate_ssm_model(model, PROBE)
    MODEL = generate_linear_model(model, signal, PROBE)
    SIGNAL = generate_measurements(signal, MODEL, sigma=1)
    tikhonov_aprox, l_curv_sol = regularized_estimation(MODEL, SIGNAL, dim=1)
    tikhonov_aprox = tikhonov_aprox.flatten()
    problem = SingleValueReconstructionProblem(MODEL, SIGNAL, l_curve_sol=None)
    return problem, PROBE, MODEL, SIGNAL, tikhonov_aprox, l_curv_sol


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


def main():
    # Configuración
    ref_point = np.array([10, 10, 10])
    archive_file = "archive/single/external_archive_pro.pickle"
    img_dir = "img/nsga2_pro/single"

    # Configurar el problema
    problem, PROBE, MODEL, SIGNAL, d_est_l_curve, l_curv_sol = setup_problem()

    # Ejecutar la optimización
    res, archive, hv_values = run_optimization(problem, ref_point, archive_file, img_dir)

    # Analizar resultados
    solutions = res.X
    objectives = res.F
    min_index = np.argmin(objectives[:, 0])
    best_solution = solutions[min_index]
    d_est_nsga2 = tikhonov(MODEL.H, SIGNAL.y, best_solution, dim=1)

    mu_est_nsga2 = mu_from_d(MODEL, d_est_nsga2)
    mu_est_l_curve = mu_from_d(MODEL, d_est_l_curve)
    
    l_curve_sol_val = problem.evaluate_l_curve_solution(l_curv_sol)


    # Graficar resultados
    plot_hypervolume(hv_values, img_dir)
    plot_pareto_front(res.F, archive=archive, img_dir=img_dir, tikhonov_sol=l_curve_sol_val)
    plot_best_solution(img_dir, PROBE, mu_est_nsga2, mu_est_l_curve)
    


if __name__ == "__main__":
    main()
