import json
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.sampling.rnd import FloatRandomSampling

from SSM.Generate_Linear_Model import generate_linear_model
from SSM.Generate_Measurements import generate_measurements
from SSM.Generate_SSM_Model import generate_ssm_model
from SSM.Set_Settings import set_settings
from SSM.Regularized_Estimation import regularized_estimation

from MO.MOP_Definition_Pymoo import ImageReconstructionProblem
from MO.custom_crossover import CustomCrossover
from MO.custom_mutation import CustomMutation
from MO.binary_tournament import binary_tournament


from utils.archive import update_archive, save_archive_pickle
from utils.plotting import plot_pareto_front, plot_best_solution


def load_archive(filename):
    """Carga un archivo externo desde un archivo JSON"""
    with open(filename, "r") as f:
        loaded_archive = json.load(f)
    loaded_solutions = [np.array(sol["X"]) for sol in loaded_archive]
    loaded_objectives = [np.array(sol["F"]) for sol in loaded_archive]
    return loaded_solutions, loaded_objectives


def setup_problem():
    """Configura el problema de optimización"""
    model, PROBE, signal = set_settings(example=1)
    model = generate_ssm_model(model, PROBE)
    MODEL = generate_linear_model(model, signal, PROBE)
    SIGNAL = generate_measurements(signal, MODEL, sigma=0)
    n_var = 20
    tikhonov_aprox = regularized_estimation(MODEL, SIGNAL, dim=1).reshape(n_var)
    problem = ImageReconstructionProblem(MODEL, PROBE, SIGNAL, n_var, tikhonov_aprox=tikhonov_aprox)
    return problem, PROBE, tikhonov_aprox


def restart_optimization(problem, loaded_solutions, archive_file, img_dir):
    """Reinicia la optimización usando soluciones del archivo externo"""
    from pymoo.core.sampling import Sampling

    class ArchiveSampling(Sampling):
        """Muestreo personalizado para inicializar la población con el archivo externo"""
        def __init__(self, solutions):
            super().__init__()
            self.solutions = solutions

        def _do(self, problem, n_samples, **kwargs):
            X = np.array(self.solutions[:n_samples])
            return X

    # Configurar el algoritmo
    algorithm = NSGA2(
        pop_size=500,
        sampling=ArchiveSampling(loaded_solutions),
        crossover=CustomCrossover(prob=0.9),
        mutation=CustomMutation(prob=0.9, max_generations=500),
        selection=TournamentSelection(func_comp=binary_tournament),
        eliminate_duplicates=True,
    )

    # Ejecutar la optimización
    res = minimize(
        problem,
        algorithm,
        seed=2,
        termination=("n_gen", 500),
        verbose=True,
    )

    # Guardar el archivo actualizado
    archive = update_archive(res.pop, [])
    save_archive_pickle(archive, archive_file)

    return res, archive


def main():
    # Configuración
    archive_file = "archive/external_archive.json"
    img_dir = "img/nsga2_pro"

    # Cargar el archivo externo
    loaded_solutions, loaded_objectives = load_archive(archive_file)

    # Configurar el problema
    problem, PROBE, tikhonov_aprox = setup_problem()

    # Analizar el archivo cargado
    print(f"Loaded solutions: {len(loaded_solutions)}")
    print(f"First solution objectives: {loaded_objectives[0]}")

    # Graficar el frente de Pareto del archivo externo
    scatter_plot = Scatter().add(np.array(loaded_objectives), label="Loaded Pareto Front")
    scatter_plot.show()
    scatter_plot.save(f"{img_dir}/loaded_pareto_front.png")

    # Reiniciar la optimización con el archivo externo
    res, archive = restart_optimization(problem, loaded_solutions, archive_file, img_dir)

    solutions = res.X
    objective_values = res.F
    min_index = np.argmin(objective_values[:, 0])
    best_solution = solutions[min_index]
    mu_est_nsga2 = problem.mo_estimation(best_solution.reshape(20, 1))
    mu_est_tikh = problem.mo_estimation(tikhonov_aprox.reshape(20, 1))

    # Graficar el frente de Pareto actualizado
    plot_pareto_front(res.F, archive, img_dir, from_loaded=True)
    plot_best_solution(img_dir, PROBE, mu_est_nsga2, mu_est_tikh, from_loaded=True)


if __name__ == "__main__":
    main()
