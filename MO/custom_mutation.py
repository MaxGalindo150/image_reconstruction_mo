import numpy as np

from pymoo.core.mutation import Mutation

class CustomMutation(Mutation):
    def __init__(self, prob=0.2, max_generations=100):
        super().__init__()
        self.prob = prob
        self.max_generations = max_generations
        self.generation = 0

    def _do(self, problem, X, **kwargs):
        self.generation += 0.1

        for i in range(X.shape[0]):  # Iterar sobre cada individuo
            if np.random.rand() < self.prob:
                mutation_magnitude = 20000 * (1 - self.generation / self.max_generations)
                for j in range(X.shape[1]):  # Iterar sobre cada gen
                    direction = np.random.uniform(0, 1)
                    if direction < 0.5:
                        X[i, j] += np.random.uniform(0, mutation_magnitude)
                    else:
                        X[i, j] -= np.random.uniform(0, mutation_magnitude)

                    # Asegurarse de que los valores mutados estén dentro de los límites
                    X[i, j] = np.clip(X[i, j], problem.xl[j], problem.xu[j])

        return X
