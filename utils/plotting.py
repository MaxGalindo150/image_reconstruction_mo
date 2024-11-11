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



def plot_pareto_front(F, archive=None, img_dir=None, tikhonov_sol=None, from_loaded=False):
    """
    Grafica el frente de Pareto y las estimaciones.

    Args:
        F (ndarray): Valores de los objetivos actuales del algoritmo.
        archive (list): Archivo externo con soluciones no dominadas.
        img_dir (str): Directorio donde guardar las imágenes.
        tikhonov_sol (ndarray): Solución obtenida con Tikhonov.
        from_loaded (bool): Si es True, indica que los datos fueron cargados.
    """
    # Graficar el frente de Pareto actual
    scatter_plot = Scatter().add(F, label="Pareto Front")
    scatter_plot.add(tikhonov_sol, label="Tikhonov Solution", color="red")
    scatter_plot.save(f"{img_dir}/pareto_front.png")

    # Graficar el frente de Pareto global (archivo externo)
    final_archive_objectives = np.array([sol["F"] for sol in archive])
    scatter_plot = Scatter().add(final_archive_objectives, label="Global Pareto Front")
    scatter_plot.add(tikhonov_sol, label="Tikhonov Solution", color="red")
    file_suffix = "_loaded" if from_loaded else ""
    scatter_plot.save(f"{img_dir}/pareto_front{file_suffix}_archive.png")


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