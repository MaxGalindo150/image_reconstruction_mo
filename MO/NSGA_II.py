import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from pymoo.operators.sampling.rnd import FloatRandomSampling
# from pymoo.util.random import set_random_seed

# pymoo_seed = 42
# numpy_seed = 42

# np.random.seed(numpy_seed)
# set_random_seed(pymoo_seed)


class Individual:
    def __init__(self, point, problem):
        self.point = point
        self.Sp = set()
        self.N = 0
        self.rank = None
        self.distance = 0
        self.values = problem.evaluate(point)


class NSGA2:
    def __init__(self, generations, population_size, mutaition_rate, problem, objective1_threshold=None):
        self.population_size = population_size
        self.mutation_rate = mutaition_rate
        self.problem = problem
        self.n_var = self.problem.n_var
        self.n_obj = self.problem.n_obj
        self.sampling = FloatRandomSampling()
        self.objective1_threshold = objective1_threshold
        self.run(generations)
        
    
    def generate_population(self):
        X = self.sampling(self.problem, self.population_size).get("X")

        return [Individual(x, self.problem) for x in X]

    
    def dominates(self, p, q):
        return all(p_i <= q_i for p_i, q_i in zip(p, q)) and any(p_i < q_i for p_i, q_i in zip(p, q))

    def non_dominated_sort(self, individuals):

        F = defaultdict(list)
        for p in individuals:
            for q in individuals:
                if p == q:
                    continue
                if self.dominates(p.values, q.values):
                    p.Sp.add(q)
                elif self.dominates(q.values, p.values):
                    p.N += 1
            if p.N == 0:
                p.rank = 1
                F[1].append(p)
        i = 1
        while F[i]:
            Q = []
            for p in F[i]:
                for q in p.Sp:
                    q.N -= 1
                    if q.N == 0:
                        q.rank = i + 1
                        Q.append(q)
            i += 1
            F[i] = Q

        return F
    
    # =======================Crowding Distance===================
    def sort_by_objective(self, front, m):
        return sorted(front, key=lambda p: p.values[m])

    def crowding_distance_assignment(self, I):
        l = len(I)
        if l == 0:
            return
        for m in range(I[0].values.size):
            I = self.sort_by_objective(I, m)
            f_min = I[0].values[m]
            f_max = I[-1].values[m]
            I[0].distance = float('inf')
            I[-1].distance = float('inf')
            for i in range(1, l - 1):
                I[i].distance += (I[i + 1].values[m] - I[i - 1].values[m])/(f_max - f_min)
        
    
        
    def sort_by_crowed_comparation(self, front):
        return sorted(front, key=lambda p: (-p.rank, p.distance), reverse=True)
    
    # =============================GA=============================
    def binary_tournament_selection(self, P):
        winners = []
        while len(winners) < len(P)//2:
            vs = np.random.choice(P, 2)
            winners.append(max(vs, key=lambda p: (-p.rank, p.distance)))
        return winners

    
    def crossover(self, individual1, individual2):
        point1 = individual1.point.copy()
        point2 = individual2.point.copy()

        # Generar un factor de mezcla aleatorio
        alpha = np.random.uniform(0, 1)

        # Realizar un cruce aritmético con el factor de mezcla aleatorio
        new_point1 = alpha * point1 + (1 - alpha) * point2
        new_point2 = (1 - alpha) * point1 + alpha * point2

        new_individual1 = Individual(new_point1, self.problem)
        new_individual2 = Individual(new_point2, self.problem)

        return new_individual1, new_individual2

    def mutate(self, individual, mutation_rate, generation, max_generations):
        point = individual.point.copy()

        # Calcular la magnitud de la mutación basada en la generación actual
        mutation_magnitude = 100 * (1 - generation / max_generations)

        # Recorrer cada elemento en el punto
        for i in range(len(point)):
            # Aplicar la mutación con una probabilidad igual a mutation_rate
            if np.random.rand() < mutation_rate:
                # Añadir un pequeño número aleatorio al elemento
                point[i] += np.random.uniform(-mutation_magnitude, mutation_magnitude)

        # Crear un nuevo individuo con el punto mutado
        point = np.array(point)
        new_individual = Individual(point, self.problem)
        return new_individual
    

    def generate_offspring(self, P, mutation_rate, generation, max_generations):
        offspring = []
        winners = self.binary_tournament_selection(P)
        while len(offspring) < len(P):
            parents = np.random.choice(winners, 2)
            offspring += self.crossover(*parents)
            offspring = [self.mutate(p, mutation_rate, generation, max_generations) for p in offspring]
        return offspring
    
    
    def initialize(self):
        self.P_t = self.generate_population()
        F = self.non_dominated_sort(self.P_t)   
        for i in range(1, len(F)):
            self.crowding_distance_assignment(F[i])
        
        self.Q_t = self.generate_offspring(self.P_t, self.mutation_rate,0,100)
        

    def run(self, generations):
        self.initialize()

        for t in tqdm(range(generations)):
            self.R_t = self.P_t + self.Q_t 
            self.F = self.non_dominated_sort(self.R_t)
            self.P_t = []
            i = 1
            while len(self.P_t) + len(self.F[i]) <= self.population_size:
                self.crowding_distance_assignment(self.F[i])
                self.P_t += self.F[i]
                i += 1
            self.sort_by_crowed_comparation(self.F[i])
            self.P_t += self.F[i][:self.population_size - len(self.P_t)]
            
            self.Q_t = self.generate_offspring(self.P_t, self.mutation_rate, t, generations)

            # check if the first objective is below the objective1_threshold
            if self.objective1_threshold is not None:
                min_obj1 = min([individual.values[0] for individual in self.P_t])
                if min_obj1 < self.objective1_threshold:
                    print(f"First objective below threshold at generation {t} value: {min_obj1} is below {self.objective1_threshold}.")
                    break


        # if self.n_obj == 2:
        #     self.plot_pareto_front()
        # elif self.n_obj == 3:
        #     self.plot_pareto_front_3d()

    def plot_pareto_front(self):
        x = [individual.values[0] for individual in self.P_t]
        y = [individual.values[1] for individual in self.P_t]

        plt.scatter(x, y, c='red')
        plt.xlabel('f1')
        plt.ylabel('f2')
        #plt.title('Pareto Front')
        plt.show()
    
    def plot_pareto_front_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = [individual.values[0] for individual in self.P_t]
        y = [individual.values[1] for individual in self.P_t]
        z = [individual.values[2] for individual in self.P_t]
        ax.scatter(x, y, z)
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        plt.show()
