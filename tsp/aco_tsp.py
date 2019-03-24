import random
import math

class AntColonyOptimization_TSP:
    @classmethod
    def __init__(self, graph, t0, m, alpha, beta, rho, delta, num_iters):
        self.graph = graph
        self.n = len(graph)  # number of nodes
        self.m = m  # number of ants
        self.pheromone = self._create_matrix(self.n, self.n, t0)
        self.distance = self._create_dist_mat(self.graph)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.delta = delta
        self.num_iters = num_iters
        self.solution = {}
        self.current_best_solution = {
            "tour": [],
            "tour_length": math.inf
        }

    @classmethod
    def run(self):
        for _ in range(self.num_iters):
            solutions = []
            for _ in range(self.m):
                solutions.append(self.ant_walk())

            lengths = []
            for solution in solutions:
                lengths.append(self.calc_tour_length(solution))

            self.evaporate_pheromone()

            best_tour_length = min(lengths)
            best_solution_index = lengths.index(best_tour_length)
            best_solution = solutions[best_solution_index]

            if best_tour_length <= self.current_best_solution["tour_length"]:
                self.current_best_solution["tour_length"] = best_tour_length
                self.current_best_solution["tour"] = best_solution

            print(self.current_best_solution)

            self.reinforce_pheromone(best_solution)
            self.reinforce_pheromone(self.current_best_solution["tour"])

    @classmethod
    def reinforce_pheromone(self, tour):
        n = len(tour)
        for i in range(n-1):
            a = tour[i]
            b = tour[i+1]
            self.pheromone[a][b] += self.delta/2

    @classmethod
    def evaporate_pheromone(self):
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= (1-self.rho)

    @classmethod
    def select_best_tour(self):
        pass

    @classmethod
    def calc_tour_length(self, tour):
        n = len(tour)
        length = 0
        for i in range(n-1):
            a = tour[i]
            b = tour[i+1]
            length += self.graph[a][b]
        return length

    @classmethod
    def ant_walk(self):
        solution = []
        nodes = [_ for _ in range(self.n)]
        start_node = random.choice(nodes)
        solution.append(start_node)
        current_node = start_node
        while len(nodes) > 1:
            nodes.remove(current_node)
            next_node = None
            while(next_node is None):
                next_node = self.choose_next_node(current_node, nodes)
            solution.append(next_node)
            current_node = next_node
        solution.append(start_node)
        return solution

    @classmethod
    def choose_next_node(self, current_node, nodes):
        nodes_copy = list(nodes)
        denominator = [self.calc_prob_numerator(
            current_node, other_node) for other_node in nodes_copy]
        denominator = sum(denominator)
        for other_node in nodes_copy:
            numerator = self.calc_prob_numerator(current_node, other_node)
            if denominator == 0:
                return other_node
            probability = numerator/denominator
            rand = random.random()
            if rand <= probability:
                return other_node
        return None

    @classmethod
    def calc_prob_numerator(self, current_node, other_node):
        numerator = self.pheromone[current_node][other_node]**self.alpha * \
            self.distance[current_node][other_node]**self.beta
        return numerator

    @staticmethod
    def _create_matrix(n, m, value=0):
        mat = []
        for _ in range(n):
            line = []
            for _ in range(m):
                line.append(value)
            mat.append(line)
        return mat

    @staticmethod
    def _print_matrix(mat, padding=1):
        n = len(mat)
        m = len(mat[0])
        if n < 1:
            return
        for i in range(n):
            line = ""
            for j in range(m):
                line += str(mat[i][j]) + padding * ' '
            print(line + '\n')

    @staticmethod
    def _create_dist_mat(graph):
        n = len(graph)
        dist_mat = AntColonyOptimization_TSP._create_matrix(n, n, 0)
        for i in range(n):
            for j in range(n):
                if graph[i][j] != 0:
                    dist_mat[i][j] = 1/graph[i][j]
        return dist_mat



    