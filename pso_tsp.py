import math
import pandas as pd
import random
from operator import attrgetter


class PSOTSP:
    def __init__(self, graph, alpha, beta, population_size, iters):
        self.graph = graph
        self.n = len(graph)
        self.alpha = alpha
        self.beta = beta
        self.g_best = math.inf
        self.g_best_sol = []
        self.population_size = population_size
        self.population = []
        self.iters = iters

    def init_population(self):
        self.population.clear()
        for _ in range(self.population_size):
            particle = self.create_rand_particle()
            self.population.append(particle)

    def velocity_selection_all_ss(self, ss_cog, ss_social):
        new_velocity = []
        if random.random() <= self.alpha:
            new_velocity += ss_cog
        if random.random() <= self.beta:
            new_velocity += ss_social
        return new_velocity
    
    def velocity_selection(self, ss_cog, ss_social):
        new_velocity = []
        for ss1 in ss_cog:
            if random.random()<= self.alpha:
                new_velocity.append(ss1)
        
        for ss2 in ss_social:
            if random.random()<= self.beta:
                new_velocity.append(ss2)
        
        return new_velocity

    def run(self):
        self.init_population()
        for _ in range(self.iters):
            self.g_best = min(self.population, key=attrgetter('cost_p_best'))

            for particle in self.population:
                particle.velocity.clear()
                particle.velocity = []
                new_velocity = []
                sol_p_best = list(particle.solution_p_best)
                sol_g_best = list(self.g_best.solution_p_best)
                sol_current = list(particle.solution_current)
                velocity_current = list(particle.velocity)

                # (pbest - x(t-1))
                ss_p_best = self.get_swap_seq(sol_p_best, sol_current)

                # (gbest - x(t-1))
                ss_g_best = self.get_swap_seq(sol_g_best, sol_current)

                new_velocity += velocity_current + ss_p_best + ss_g_best

                particle.velocity = new_velocity

                temp_velocity = self.velocity_selection(ss_p_best, ss_g_best)
                
                new_solution = self.make_swap_seq(sol_current, temp_velocity)
                new_cost = self.calc_cost(new_solution)

                particle.solution_current = new_solution
                particle.cost_current = new_cost                

                # Update pbest
                if particle.cost_current < particle.cost_p_best:
                    particle.cost_p_best = particle.cost_current
                    particle.solution_p_best = particle.solution_current

            self.print_best_tour()


    def create_rand_particle(self):
        solution = self.create_rand_solution()
        cost = self.calc_cost(solution)
        particle = Particle(solution, cost)
        return particle

    def get_swap_seq(self, sol_a, sol_b):
        temp_sol = list(sol_b)
        ss = []
        for i in range(self.n):
            if sol_a[i] != temp_sol[i]:
                so = (i, temp_sol.index(sol_a[i]))
                ss.append(so)
                temp_sol = self.make_swap(sol_b, so)

        return ss

    def make_swap(self, sol_x, so):
        new_sol = list(sol_x)  # Copy input solution
        a, b = so
        temp = new_sol[b]
        new_sol[b] = new_sol[a]
        new_sol[a] = temp
        return new_sol

    def make_swap_seq(self, sol_x, ss):
        new_sol = list(sol_x)
        for so in ss:
            new_sol = self.make_swap(new_sol, so)
        return new_sol

    def create_rand_solution(self):
        tour = [_ for _ in range(self.n)]
        random.shuffle(tour)
        return tour

    def calc_cost(self, sol):
        distance = 0
        for i in range(self.n - 1):
            a = sol[i]
            b = sol[i + 1]
            distance += self.graph[a][b]
        a = sol[-1]
        b = sol[0]
        distance += self.graph[a][b]
        return distance

    def print_best_tour(self):
        tour_info = {'tour':self.g_best.solution_p_best, 'tour_length':self.g_best.cost_p_best}
        print(tour_info)

class Particle:
    def __init__(self, solution, cost):
        self.solution_current = solution
        self.cost_current = cost
        self.solution_p_best = solution
        self.cost_p_best = cost
        self.velocity = []

    def __str__(self):
        string = ""
        string += "P-best solution: " + str(self.solution_p_best) + "\n"
        string += "P-best cost: " + str(self.cost_p_best) + "\n"
        string += "Current solution: " + str(self.solution_current) + "\n"
        string += "Current solution cost: " + str(self.cost_current) + "\n"
        string += "Current velocity: " + str(self.velocity) + "\n"
        return string
    
    
        


if __name__ == "__main__":
    full_path = "/Users/pelanmar1/Coding/Tesis/heuristics/testdata.xlsx"
    # Load a dataset
    df = pd.read_excel(full_path, sheet_name="ATT48", header=None)
    graph = df.as_matrix()
    random.seed(100)
    alpha = 0.8
    beta = 0.8
    population_size = 50
    iters = 100
    pso = PSOTSP(graph, alpha, beta, population_size, iters)
    pso.run()
