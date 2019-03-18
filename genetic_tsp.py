from random import shuffle, uniform, randint
import math
from tsp_plotter import TSPPlotter
class GeneticTSP:
    def __init__(self, graph_matrix, elitism=True, tournament_size=5, mutation_rate=0.015, num_generations=1000, population_size=200):
        self.graph_matrix = graph_matrix
        self.genome_length = len(graph_matrix)
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.num_generations = num_generations
        self.population_size = population_size
        self.current_best_solution = {
            "tour": [],
            "tour_length": math.inf
        }
        self.run_data = []

    def _update_best_solution(self, tour, tour_length):
        self.current_best_solution["tour"] = tour
        self.current_best_solution["tour_length"] = tour_length

    def update_run_data(self):
        self.run_data.append(self.current_best_solution)


    def create_individual(self):
        individual = [i for i in range(self.genome_length)]
        shuffle(individual)
        return individual

    def evaluate_individual(self, indiviudal):
        accum_distance = 0
        for i in range(self.genome_length-1):
            a = indiviudal[i]
            b = indiviudal[i+1]
            accum_distance += self.graph_matrix[a][b]
        last_city = indiviudal[-1]
        origin = indiviudal[0]
        accum_distance += self.graph_matrix[last_city][origin]
        return accum_distance

    def create_population(self, population_size):
        population = []
        for _ in range(population_size):
            new_individual = self.create_individual()
            population.append(new_individual)
        return population

    def get_fittest(self, population, minimize=True):
        population_size = len(population)
        fittest = 0
        for i in range(population_size):
            score = self.evaluate_individual(population[i])
            if score <= self.evaluate_individual(population[fittest]):
                if minimize:
                    fittest = i
            else:
                if not minimize:
                    fittest = i
        return population[fittest]

    def mutate(self, individual):
        for i in range(len(individual)):
            if uniform(0, 1) < self.mutation_rate:
                swap_index = randint(0, len(individual) - 1)
                temp = individual[i]
                individual[i] = individual[swap_index]
                individual[swap_index] = temp
        return individual

    def crossover(self, individual1, individual2):
        new_individual = [0 for i in range(len(individual1))]
        indx_1 = randint(0, len(individual1) - 1)
        indx_2 = randint(0, len(individual1) - 1)
        start_pos = min(indx_1, indx_2)
        end_pos = max(indx_1, indx_2)
        temp = individual1[start_pos:(end_pos+1)]
        new_individual[start_pos:(end_pos+1)] = temp
        pos = 0
        for i in range(len(new_individual)):
            while start_pos <= pos <= end_pos:
                pos += 1
            if individual2[i] not in temp:
                new_individual[pos] = individual2[i]
                pos += 1
        return new_individual

    def evolve(self, population):
        new_population = []
        elite_offset = 0
        if self.elitism:
            elite = self.get_fittest(population)
            new_population.insert(0, elite)
            elite_offset = 1
        for i in range(elite_offset, len(population)):
            parent1 = self.select_candidate(population)
            parent2 = self.select_candidate(population)
            new_individual = self.crossover(parent1, parent2)
            new_population.insert(i, new_individual)

        for i in range(elite_offset, len(population)):
            mutated_individual = self.mutate(new_population[i])
            new_population[i] = mutated_individual
        return new_population

    def select_candidate(self, population):
        tournament = []
        for _ in range(self.tournament_size):
            rand_index = randint(0, len(population) - 1)
            tournament.append(population[rand_index])
        fittest = self.get_fittest(tournament)
        return fittest

    def run(self):
        population = self.create_population(self.population_size)
        for _ in range(self.num_generations):
            population = self.evolve(population)
            fittest = self.get_fittest(population)
            score = self.evaluate_individual(fittest)
            self._update_best_solution(fittest, score)
            print(self.current_best_solution)
            self.update_run_data()

        return self.run_data
