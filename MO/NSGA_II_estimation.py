from MO.MOP_Definition import ImageReconstructionProblem
from MO.NSGA_II import NSGA2

def nsga2_estimation(MODEL, PROBE, SIGNAL, n_var, b):

    PROBLEM = ImageReconstructionProblem(MODEL, PROBE, SIGNAL, n_var, b)
    nsga2 = NSGA2(generations=500, population_size=100, mutation_rate=0.8, problem=PROBLEM, objective1_threshold=0.01)
    x = [individual.values[0] for individual in nsga2.P_t]
    y = [individual.values[1] for individual in nsga2.P_t]
    best_individual_nsga2 = min(nsga2.P_t, key=lambda p: p.values[0])
    d_est_nsga2 = best_individual_nsga2.point.reshape(20, 1)

    #mu_est_nsga2 = PROBLEM.mo_estimation(d_est_nsga2)

    return d_est_nsga2