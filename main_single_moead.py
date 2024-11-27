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
    run_optimization_moead,
    get_best_solution,
    save_solutions_to_csv
)


def main():
    ref_point = np.array([10, 10, 10])  # Punto de referencia para el cálculo de hipervolumen
    archive_file = "archive/single/moead/external_archive_pro.pickle"
    img_dir = "img/nsga2_pro/single/moead"

    # Configurar el problema
    problem, PROBE, MODEL, SIGNAL, d_lcurve, lambda_lcurve = setup_problem(example=1, sigma=0.1)

    # Ejecutar la optimización
    res, archive, hv_values = run_optimization_moead(problem, ref_point, archive_file, img_dir)

    solutions = res.X
    objectives = res.F
    
    # Colorear por puntajes ponderados
    weights = [0.6, 0.3, 0.1]  # Pesos personalizados para los objetivos
    
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
    output_csv_file = "solutions_comparison_moead.csv"
    save_solutions_to_csv(l_curve_sol_values, lambda_lcurve, best_solution_values, best_index, objectives, solutions, output_csv_file)

    # Graficar resultados principales
    plot_hypervolume(hv_values, img_dir, "MOEA/D")
    plot_pareto_front(
        res.F, "MOEA/D",archive=archive, img_dir=img_dir,
        lcurv_sol=l_curve_sol_values, best_sol=best_solution_values
    )
    plot_best_solution(img_dir, PROBE, mu_est_nsga2, mu_est_l_curve, "MOEA/D")
    
    # Graficar análisis adicional del frente de Pareto
    # Visualización con clustering
    visualize_pareto_front(objectives, img_dir, "MOEA/D", clusters=3)

    # Proyecciones 2D
    visualize_pareto_2d_projections(objectives, img_dir, "MOEA/D")

    
    visualize_pareto_with_weights(objectives, weights, img_dir, "MOEA/D")


if __name__ == "__main__":
    main()
