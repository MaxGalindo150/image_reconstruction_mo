import numpy as np

from utils.plotting import plot_hypervolume, plot_pareto_front, plot_best_solution
from utils.optimization import (
    setup_problem,
    run_optimization,
    get_best_solution,
    save_solutions_to_csv
)



def main():
    ref_point = np.array([10, 10, 10])
    archive_file = "archive/single/external_archive_pro.pickle"
    img_dir = "img/nsga2_pro/single"

    # Configurar el problema
    problem, PROBE, MODEL, SIGNAL, d_lcurve, lambda_lcurve = setup_problem(example=1, sigma=0.05)

    # Ejecutar la optimización
    res, archive, hv_values = run_optimization(problem, ref_point, archive_file, img_dir)

    solutions = res.X
    objectives = res.F
    
    # Seleccionar la mejor solución basada en ponderaciones
    l_curve_sol_values, best_solution_values, mu_est_l_curve, mu_est_nsga2, best_index = get_best_solution(objectives=objectives,                                                                              solutions=solutions,                                                                                   d_lcurve=d_lcurve,
    lambda_lcurve=lambda_lcurve,problem=problem,                                                                                     MODEL=MODEL,                                                                                              SIGNAL=SIGNAL)
    

    # Guardar las soluciones en un archivo CSV
    output_csv_file = "solutions_comparison.csv"
    save_solutions_to_csv(l_curve_sol_values, lambda_lcurve, best_solution_values, best_index, objectives, solutions, output_csv_file)


    # Graficar resultados
    plot_hypervolume(hv_values, img_dir)
    plot_pareto_front(res.F, archive=archive, img_dir=img_dir, lcurv_sol=l_curve_sol_values, best_sol=best_solution_values)
    plot_best_solution(img_dir, PROBE, mu_est_nsga2, mu_est_l_curve)
    


if __name__ == "__main__":
    main()
