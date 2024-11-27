import numpy as np

from utils.plotting import (
    plot_hypervolume, 
    plot_pareto_front, 
    plot_best_solution,
    visualize_pareto_2d_projections,
    visualize_pareto_front,
    visualize_pareto_with_weights
) 

    
from utils.optimization import (
    setup_problem,
    run_optimization_nsga3,
    get_best_solution,
    save_solutions_to_csv
)


def main():
    ref_point = np.array([10, 10, 10])  # Punto de referencia para el cálculo de hipervolumen
    archive_file = "archive/single/external_archive_pro.pickle"
    img_dir = "img/nsga2_pro/single/nsga3"

    # Configurar el problema
    problem, PROBE, MODEL, SIGNAL, d_lcurve, lambda_lcurve = setup_problem(example=1, sigma=0.1)

    # Ejecutar la optimización
    res, archive, hv_values = run_optimization_nsga3(problem, ref_point, archive_file, img_dir)

    solutions = res.X
    objectives = res.F
    
    # Colorear por puntajes ponderados
    weights = [0.6, 0.1, 0.3]  # Pesos personalizados para los objetivos
    
    # Seleccionar la mejor solución basada en ponderaciones
    l_curve_sol_values, best_solution_values, mu_est_l_curve, mu_est_nsga2, best_index = get_best_solution(
        objectives=objectives,
        solutions=solutions,
        d_lcurve=d_lcurve,
        lambda_lcurve=lambda_lcurve,
        problem=problem,
        MODEL=MODEL,
        SIGNAL=SIGNAL,
        weights=weights
    )
    
    # Guardar las soluciones en un archivo CSV
    output_csv_file = "solutions_comparison_nsga3.csv"
    save_solutions_to_csv(l_curve_sol_values, lambda_lcurve, best_solution_values, best_index, objectives, solutions, output_csv_file)

    # Graficar resultados principales
    plot_hypervolume(hv_values, img_dir, "NSGA-III")
    plot_pareto_front(
        res.F, "NSGA-III",archive=archive, img_dir=img_dir,
        lcurv_sol=l_curve_sol_values, best_sol=best_solution_values
    )
    plot_best_solution(img_dir, PROBE, mu_est_nsga2, mu_est_l_curve, "NSGA-III")
    
    # Graficar análisis adicional del frente de Pareto
    # Visualización con clustering
    visualize_pareto_front(objectives, img_dir, "NSGA-III", clusters=4)

    # Proyecciones 2D
    visualize_pareto_2d_projections(objectives, img_dir, "NSGA-III")

    
    visualize_pareto_with_weights(objectives, weights, img_dir, "NSGA-III")


if __name__ == "__main__":
    main()
