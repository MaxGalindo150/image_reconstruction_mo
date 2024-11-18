import matplotlib.pyplot as plt
import numpy as np
from pymoo.visualization.scatter import Scatter


def plot_hypervolume(hv_values, img_dir, from_loaded=False):
    """Grafica el hipervolumen a lo largo de las generaciones"""
    plt.figure(figsize=(10, 6))
    plt.plot(hv_values, label="Hypervolume")
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Hypervolume", fontsize=14)
    plt.title("Hypervolume over Generations", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    if from_loaded:
        plt.savefig(f"{img_dir}/hypervolume_loaded.png")
    else:
        plt.savefig(f"{img_dir}/hypervolume.png")



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


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

    # Estilo para el frente de Pareto
    ax.scatter(
        F[:, 0], F[:, 1], F[:, 2],
        c="blue", label="Pareto Front", alpha=0.3, marker="o", s=50
    )

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
    """Grafica la mejor solución encontrada y las estimaciones"""
    plt.figure(figsize=(10, 6))
    plt.plot(mu_est_nsga2, label='NSGA-II Estimation', linestyle='-', color='b')
    plt.plot(PROBE.mu, label='True mu', linestyle='--', color='g')
    plt.plot(mu_est_tikh, label='Tikhonov Estimation', linestyle='-.', color='r')
    plt.xlabel('Depth', fontsize=14)
    plt.ylabel('mu', fontsize=14)
    plt.title('Sin informar a NSGA-II (pymoo)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    if from_loaded:
        plt.savefig(f"{img_dir}/best_solution_loaded.png")
    else:
        plt.savefig(f"{img_dir}/best_solution.png")