import random
import math

class SimulatedAnnealingTSP:
    @classmethod
    def __init__(self, temperature, cooling_rate, graph, initial_tour=None):
        self.graph = graph
        self.n = len(graph)
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.initial_tour = initial_tour
        self.current_best_solution = {
            "tour": [],
            "tour_length": math.inf
        }

    @classmethod
    def _generate_tour(self):
        tour = [_ for _ in range(self.n)]
        random.shuffle(tour)
        return tour

    @classmethod
    def _get_distance(self, tour):
        distance = 0
        for i in range(self.n - 1):
            a = tour[i]
            b = tour[i + 1]
            distance += self.graph[a][b]
        a = tour[-1]
        b = tour[0]
        distance += self.graph[a][b]
        return distance

    @classmethod
    def _create_neighbour(self, tour):
        new_tour = list(tour)
        city_swap_1 = random.randint(0, self.n - 1)
        city_swap_2 = random.randint(0, self.n - 1)
        temp = tour[city_swap_1]
        new_tour[city_swap_1] = tour[city_swap_2]
        new_tour[city_swap_2] = temp
        return new_tour

    @classmethod
    def _acceptance_probability(self, energy, new_energy, temperature):
        if new_energy < energy:
            return 1.0
        delta_energy = energy - new_energy
        return math.exp(delta_energy/temperature)

    @classmethod
    def _update_best_solution(self, tour, tour_length):
        self.current_best_solution["tour"] = tour
        self.current_best_solution["tour_length"] = tour_length

    @classmethod
    def run(self):
        if self.initial_tour:
            current_solution = self.initial_tour
        else:    
            current_solution = self._generate_tour()
        best_tour = current_solution
        best_tour_distance = str(self._get_distance(best_tour))
        print(self.current_best_solution)

        while self.temperature > 1:
            new_solution = self._create_neighbour(current_solution)
            current_energy = self._get_distance(current_solution)
            neighbour_energy = self._get_distance(new_solution)

            probability = self._acceptance_probability(
                current_energy, neighbour_energy, self.temperature)

            if random.random() <= probability:
                current_solution = new_solution

            if self._get_distance(current_solution) < self._get_distance(best_tour):
                best_tour = current_solution
                best_tour_distance = self._get_distance(best_tour)
                self._update_best_solution(best_tour, best_tour_distance)

            self.temperature *= (1 - self.cooling_rate)

            print(self.current_best_solution)
