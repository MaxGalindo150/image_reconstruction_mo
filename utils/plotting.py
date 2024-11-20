import matplotlib.pyplot as plt
import numpy as np
from pymoo.visualization.scatter import Scatter


def plot_hypervolume(hv_values, img_dir, from_loaded=False):
    """Graph the hypervolume convergence across generations"""
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Graficar hipervolumen
    plt.plot(
        hv_values, 
        label="Hypervolume", 
        linestyle='-', 
        color='#1f77b4', 
        linewidth=2, 
        marker='o', 
        markersize=6, 
        markerfacecolor='#ff7f0e', 
        markeredgewidth=2
    )
    
    # Etiquetas de los ejes
    plt.xlabel("Generation", fontsize=16, labelpad=10)
    plt.ylabel("Hypervolume", fontsize=16, labelpad=10)
    
    # Título de la gráfica
    plt.title("Hypervolume Convergence Across Generations", fontsize=18, pad=20, fontweight='bold')
    
    # Leyenda estilizada
    plt.legend(fontsize=14, loc='lower right', frameon=True, shadow=True, borderpad=1)
    
    # Cuadrícula mejorada
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Ajustar límites para mayor claridad
    plt.xlim([0, len(hv_values) - 1])
    plt.ylim([min(hv_values) - 0.01 * (max(hv_values) - min(hv_values)), max(hv_values) + 0.01 * (max(hv_values) - min(hv_values))])
    
    # Ajuste de bordes para evitar recortes
    plt.tight_layout()
    
    # Guardar la figura
    file_name = "hypervolume_loaded.png" if from_loaded else "hypervolume.png"
    plt.savefig(f"{img_dir}/{file_name}", dpi=300, bbox_inches="tight")





def plot_pareto_front(F, archive=None, img_dir=None, lcurv_sol=None, best_sol=None):
    """
    Grafica el frente de Pareto, la solución L-Curve y la mejor solución seleccionada con matplotlib.

    Args:
        F (ndarray): Valores de los objetivos actuales del algoritmo.
        archive (list): Archivo externo con soluciones no dominadas.
        img_dir (str): Directorio donde guardar las imágenes.
        lcurv_sol (ndarray): Solución obtenida con Tikhonov.
        best_sol (ndarray): Mejor solución seleccionada en el frente de Pareto.
    """
    # Preparar figura en 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', azim=45, elev=30)
    
    # Agregar solución de la curva L
    if lcurv_sol is not None:
        lcurv_sol = lcurv_sol.reshape(1, -1)
        ax.scatter(
            lcurv_sol[0, 0], lcurv_sol[0, 1], lcurv_sol[0, 2],
            c="red", label="L-Curve Solution", marker="*", s=200, edgecolor="k"
        )

    # Agregar la mejor solución seleccionada
    if best_sol is not None:
        best_sol = best_sol.reshape(1, -1)
        ax.scatter(
            best_sol[0, 0], best_sol[0, 1], best_sol[0, 2],
            c="green", label="Best NSGA-II Solution", marker="^", s=200, edgecolor="k"
        )

    # Estilo para el frente de Pareto
    ax.scatter(
        F[:, 0], F[:, 1], F[:, 2],
        c="blue", label="Pareto Front", alpha=0.3, marker="o", s=50
    )

    


    # Configurar etiquetas de los ejes
    ax.set_xlabel("f1 (Residual)", fontsize=12, labelpad=15)
    ax.set_ylabel("f2 (Regularization)", fontsize=12, labelpad=15)
    ax.set_zlabel("f3 (Negativity Penalty)", fontsize=12, labelpad=15)
    ax.set_title("Pareto Front with L-Curve and Best NSGA-II Solution", fontsize=14, fontweight="bold", pad=20)

    # Ajustar límites para una mejor visualización
    ax.set_xlim([F[:, 0].min() - 0.1, F[:, 0].max() + 0.1])
    ax.set_ylim([F[:, 1].min() - 0.1, F[:, 1].max() + 0.1])
    ax.set_zlim([F[:, 2].min() - 0.1, F[:, 2].max() + 0.1])

    # Agregar una cuadrícula y leyenda
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=12, loc="upper right")

    # Guardar la figura
    if img_dir is not None:
        plt.savefig(f"{img_dir}/pareto_front_with_best_solution.png", dpi=300, bbox_inches="tight")


def plot_best_solution(img_dir, PROBE, mu_est_nsga2, mu_est_tikh, from_loaded=False):
    """Grafica la mejor solución encontrada y las estimaciones con un estilo profesional."""
    
    # Crear figura con estilo
    plt.figure(figsize=(16, 12))
    plt.plot(mu_est_nsga2, label='NSGA-II Estimation', linestyle='-', color='#1f77b4', linewidth=4)
    plt.plot(PROBE.mu, label='True $\mu$', linestyle='--', color='#d62728', linewidth=4)
    plt.plot(mu_est_tikh, label='Tikhonov Estimation', linestyle='-.', color='#9467bd', linewidth=4)
    
    # Etiquetas de los ejes
    plt.xlabel('Depth (mm)', fontsize=16, labelpad=10)
    plt.ylabel(r'$\mu$ (Absorption)', fontsize=16, labelpad=10)
    
    # Título de la gráfica
    plt.title('Comparison of Estimations: True vs. NSGA-II vs. Tikhonov', fontsize=18, pad=20, fontweight='bold')
    
    # Leyenda mejorada
    legend = plt.legend(fontsize=14, loc='upper right', frameon=True, shadow=True, borderpad=1)
    legend.get_frame().set_alpha(0.7)
    
    # Cuadrícula más estética
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Ajuste de bordes para evitar recortes
    plt.tight_layout()
    
    # Guardar la figura
    file_name = "best_solution_loaded.png" if from_loaded else "best_solution.png"
    plt.savefig(f"{img_dir}/{file_name}", dpi=300, bbox_inches="tight")
    
def plot_rmse_vs_noise(sigmas, avg_rmse_lsq, avg_rmse_nsga2, avg_rmse_tikhonov, img_dir="img"):
    """Grafica el impacto del ruido en el RMSE con estilo profesional."""
    
    plt.figure(figsize=(10, 7))
    
    # Graficar los valores con marcadores estilizados
    plt.plot(sigmas, avg_rmse_lsq, label='LSQ', marker='o', markersize=8, linestyle='-', color='#1f77b4', linewidth=2)
    plt.plot(sigmas, avg_rmse_nsga2, label='NSGA-II', marker='x', markersize=8, linestyle='--', color='#d62728', linewidth=2)
    plt.plot(sigmas, avg_rmse_tikhonov, label='L-Curve', marker='s', markersize=8, linestyle='-.', color='#2ca02c', linewidth=2)
    
    # Escalas logarítmicas
    plt.xscale('log')
    plt.yscale('log')
    
    # Etiquetas de los ejes
    plt.xlabel(r'Noise Variance ($\sigma_w^2$)', fontsize=14, labelpad=10)
    plt.ylabel('Average RMSE', fontsize=14, labelpad=10)
    
    # Título
    plt.title('Impact of Noise on Average RMSE', fontsize=16, pad=15, fontweight='bold')
    
    # Leyenda estilizada
    plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True, borderpad=1)
    
    # Cuadrícula mejorada
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Ajuste de los márgenes
    plt.tight_layout()
    
    # Guardar la gráfica
    plt.savefig(f"{img_dir}/impact_of_noise_on_rmse_single.png", dpi=300, bbox_inches="tight")
    plt.show()