import json
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import pickle  # Para almacenamiento binario opcional


def dominates(F1, F2):
    """Verifica si F1 domina a F2"""
    return np.all(F1 <= F2) and np.any(F1 < F2)


def is_non_dominated(solution, archive):
    """Verifica si una solución es no dominada en comparación con el archivo."""
    dominated_solutions = []  # Lista para almacenar soluciones dominadas

    for existing_solution in archive:
        if dominates(existing_solution["F"], solution["F"]):
            return False  # La solución es dominada
        if dominates(solution["F"], existing_solution["F"]):
            dominated_solutions.append(existing_solution)  # Marcar solución para eliminación

    archive[:] = [sol for sol in archive if not any(np.array_equal(sol["F"], ds["F"]) for ds in dominated_solutions)]

    return True



from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def update_archive(pop, archive):
    """Actualiza el archivo externo utilizando NonDominatedSorting."""
    # Obtener los valores objetivos y las soluciones del archivo
    F_archive = np.array([sol["F"] for sol in archive])
    X_archive = np.array([sol["X"] for sol in archive])

    # Obtener los valores objetivos y las soluciones de la población actual
    F_pop = pop.get("F")
    X_pop = pop.get("X")

    # Combinar las soluciones del archivo con las nuevas soluciones
    F_combined = np.vstack([F_archive, F_pop]) if len(F_archive) > 0 else F_pop
    X_combined = np.vstack([X_archive, X_pop]) if len(X_archive) > 0 else X_pop

    # Ordenar las soluciones combinadas para obtener el frente de Pareto
    front_indices = NonDominatedSorting().do(F_combined, only_non_dominated_front=True)

    # Actualizar el archivo con las soluciones no dominadas
    updated_archive = [{"X": X_combined[i], "F": F_combined[i]} for i in front_indices]
    return updated_archive



def save_archive_json(archive, filename):
    """Guarda el archivo externo en formato JSON"""
    with open(filename, "w") as f:
        json.dump([{"X": sol["X"].tolist(), "F": sol["F"].tolist()} for sol in archive], f)


def save_archive_pickle(archive, filename):
    """Guarda el archivo externo en formato binario (pickle)"""
    with open(filename, "wb") as f:
        pickle.dump(archive, f)


def load_archive_json(filename):
    """Carga el archivo externo desde un archivo JSON"""
    with open(filename, "r") as f:
        data = json.load(f)
    return [{"X": np.array(sol["X"]), "F": np.array(sol["F"])} for sol in data]


def load_archive_pickle(filename):
    """Carga el archivo externo desde un archivo binario (pickle)"""
    with open(filename, "rb") as f:
        return pickle.load(f)


def limit_archive_size(archive, max_size):
    """
    Limita el tamaño del archivo externo a max_size soluciones.
    
    Args:
        archive (list): Lista de soluciones (diccionarios con "X" y "F").
        max_size (int): Número máximo de soluciones a mantener en el archivo.

    Returns:
        list: Archivo externo limitado al tamaño especificado.
    """
    if len(archive) <= max_size:
        return archive  # No es necesario limitar

    # Ordenar el archivo por la suma de los valores objetivos (como criterio simplificado)
    archive.sort(key=lambda sol: np.sum(sol["F"]))

    # Mantener solo las primeras max_size soluciones
    return archive[:max_size]
