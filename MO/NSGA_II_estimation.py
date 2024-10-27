import numpy as np

from MO.MOP_Definition_Pymoo import ImageReconstructionProblem
from MO.NSGA_II import NSGA2
from External_lib.Tikhonov import tikhonov

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter



def nsga2_estimation(MODEL, PROBE, SIGNAL, n_var, b, lambda_tikhonov):

    # Tikhonov
    tikhonov_aprox = tikhonov(MODEL.H, b, lambda_tikhonov, dim=2) #.reshape(1, n_var)

    # problem = ImageReconstructionProblem(MODEL, PROBE, SIGNAL, n_var, b, tikhonov_aprox = tikhonov_aprox)
    


    # # Crear las direcciones de referencia para la optimización
    # ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)

    # # Crear el objeto del algoritmo
    # algorithm = NSGA3(pop_size=20, ref_dirs=ref_dirs)

    # # Ejecutar la optimización
    # res = minimize(problem,
    #             algorithm,
    #             seed=1,
    #             termination=('n_gen', 600))

    # # Acceder a las soluciones y valores de las funciones objetivo
    # solutions = res.X
    # objective_values = res.F

    # # Encontrar el índice de la solución con el menor valor en el objetivo 1
    # min_index = np.argmin(objective_values[:, 0])

    # # Obtener la solución correspondiente a ese índice
    # best_solution = solutions[min_index]
    # best_objective_values = objective_values[min_index]
    

    return tikhonov_aprox
#best_solution