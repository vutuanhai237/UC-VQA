import types
import qiskit
from qtm.evolution import ecircuit
import qtm.random_circuit
import qtm.state
import qtm.qcompilation
import qtm.ansatz
import random
import numpy as np
import matplotlib.pyplot as plt
import qtm.progress_bar


class EEnvironment():
    def __init__(self, params: [],
                 fitness_func: types.FunctionType,
                 crossover_func: types.FunctionType,
                 mutate_func: types.FunctionType,
                 selection_func: types.FunctionType,
                 pool) -> None:
        self.best_score_progress = []
        self.scores_in_loop = []
        self.depth = params['depth']
        self.num_individual = params['num_individual']  # Must mod 8 = 0
        self.num_generation = params['num_generation']
        self.num_qubits = params['num_qubits']
        self.prob_mutate = params['prob_mutate']
        self.threshold = params['threshold']
        self.fitness_func = fitness_func
        self.crossover_func = crossover_func
        self.mutate_func = mutate_func
        self.selection_func = selection_func
        self.fitness = 0
        self.pool = pool
        self.best_candidate = None
        self.population = []
        return

    def evol(self, verbose: int = 1):
        if verbose == 1:
            bar = qtm.progress_bar.ProgressBar(
                max_value=self.num_generation, disable=False)
        for generation in range(self.num_generation):
            self.scores_in_loop = []
            new_population = []
            # Selection
            self.population = self.selection_func(self.population)
            for i in range(0, self.num_individual, 2):
                # Crossover
                child_1, child_2 = self.crossover_func(
                    self.population[i], self.population[i+1])
                new_population.extend([child_1, child_2])
                self.scores_in_loop.extend([child_1.fitness, child_2.fitness])
            self.population = new_population
            # Mutate
            for individual in self.population:
                if random.random() < self.prob_mutate:
                    self.mutate_func(individual, self.pool)

            best_score = np.min(self.scores_in_loop)
            best_index = np.argmin(self.scores_in_loop)
            if self.best_candidate.fitness > self.population[best_index].fitness:
                self.best_candidate = self.population[best_index]
            self.best_score_progress.append(best_score)
            if verbose == 1:
                bar.update(1)
            if verbose == 2 and generation % 5 == 0:
                print("Step " + str(generation) + ": " + str(best_score))
            if self.threshold(best_score):
                break
        print('End best score, end evol progress, percent target: %.1f' % best_score)
        return

    def initialize_population(self):
        self.population = []
        for _ in range(self.num_individual):
            random_circuit = qtm.random_circuit.generate_with_pool(
                self.num_qubits, self.depth, self.pool)
            individual = ecircuit.ECircuit(
                random_circuit,
                self.fitness_func)
            individual.compile()
            self.population.append(individual)
        self.best_candidate = self.population[0]
        return

    def plot(self):
        plt.plot(self.best_score_progress)
        plt.xlabel('No. generation')
        plt.ylabel('Best score')
        plt.show()
