import random
import itertools
import numpy as np
import math
import pandas as pd


class TabuSearchTSP:

    @classmethod
    def __init__(self, graph, iters, tabu_k):
        self.graph = graph
        self.n = len(graph)
        self.tabu_struct = self._create_matrix(self.n, self.n, 0)
        self.iters = iters
        self.tabu_k = tabu_k
        self.current_best_solution = {
            "tour": [],
            "tour_length": math.inf
        }

    @classmethod
    def _update_best_solution(self, tour, tour_length):
        self.current_best_solution["tour"] = tour
        self.current_best_solution["tour_length"] = tour_length

    @classmethod
    def create_individual(self):
        tour = [_ for _ in range(self.n)]
        # random.shuffle(tour)
        return tour

    @classmethod
    def _create_matrix(self, n, m, k):
        matrix = []
        for _ in range(n):
            row = []
            for i in range(m):
                row.append(k)
            matrix.append(row)
        return matrix

    @classmethod
    def decrement_tabo(self):
        n = len(self.tabu_struct)
        for i in range(n):
            for j in range(n):
                if self.tabu_struct[i][j] > 0:
                    self.tabu_struct[i][j] -= 1

    @classmethod
    def tabu_move(self, a, b):
        step = self.tabu_k
        self.tabu_struct[a][b] += step
        self.tabu_struct[b][a] += step

    @classmethod
    def get_best_neighbor(self, tour):
        current_best_tour = tour
        current_best_cost = self.evaluate_solution(tour)
        current_best_move = None
        pairs = list(itertools.combinations(tour[1:], 2))
        for p in pairs:
            neighbor = list(tour)
            a = p[0]
            b = p[1]
            neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
            new_cost = self.evaluate_solution(neighbor)
            if new_cost < current_best_cost and self.tabu_struct[a][b] == 0:
                current_best_tour = neighbor
                current_best_cost = new_cost
                current_best_move = p

        self.decrement_tabo()
        if current_best_move:
            self.tabu_move(current_best_move[0], current_best_move[1])
        current_best_solution = {
            "tour": current_best_tour, "tourlength": current_best_cost}
        return current_best_solution

    @classmethod
    def evaluate_solution(self, tour):
        distance = 0
        for i in range(len(tour)-1):
            a = tour[i]
            b = tour[i+1]
            d = self.graph[a][b]
            distance += d
        last = tour[-1]
        origin = tour[0]
        distance += self.graph[last][origin]
        return distance

    @classmethod
    def run(self):
        current_tour = self.create_individual()
        current_cost = self.evaluate_solution(current_tour)
        self._update_best_solution(current_tour, current_cost)
        for _ in range(self.iters):
            print(self.current_best_solution)
            new_solution = self.get_best_neighbor(current_tour)
            new_cost = new_solution["tourlength"]
            new_tour = new_solution["tour"]
            if new_cost <= current_cost:
                self._update_best_solution(new_tour, new_cost)
                current_tour = new_tour
                current_cost = new_cost
