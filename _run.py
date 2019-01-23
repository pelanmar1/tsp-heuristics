import random
import math
import pandas as pd
# pip install xlrd
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from genetic_tsp import GeneticTSP
from aco_tsp import AntColonyOptimization_TSP
from sa_tsp import SimulatedAnnealingTSP
from tabu_tsp import TabuSearchTSP

'''
    - ATT48 is a set of 48 cities (US state capitals) from TSPLIB. The minimal tour has length 10628.

    - DANTZIG42 is a set of 42 cities, from TSPLIB. The minimal tour has length 699.
      (Almost Solved (length of 700) <-> t0=1, m=1000, alpha=1, beta=5,rho=0.5, delta=0.5, num_iters=50)

    - FRI26 is a set of 26 cities, from TSPLIB. The minimal tour has length 937.
      (Solved <-> t0=1, m=1000, alpha=1, beta=1,rho=0.3, delta=0.3, num_iters=50)

    - FIVE is a set of 5 cities. The minimal tour has length 19.
      (Solved <-> t0=1, m=1000, alpha=1, beta=5,rho=0.5, delta=0.9, num_iters=50)

    - GR17 is a set of 17 cities, from TSPLIB. The minimal tour has length 2085.
      (Solved <-> t0=1, m=1000, alpha=1, beta=5,rho=0.5, delta=0.9, num_iters=50)

    - P01 is a set of 15 cities. It is NOT from TSPLIB. The minimal tour has length 291.
      (Solved <-> t0=1, m=1000, alpha=1, beta=5,rho=0.5, delta=0.9, num_iters=50)

    - TESTGOOGLE minimal length is 7293
      (Solved <-> t0=1, m=1000, alpha=1, beta=5,rho=0.5, delta=0.9, num_iters=50)

'''
full_path = "/Users/pelanmar1/Coding/Tesis/heuristics/testdata.xlsx"
# Load a dataset
df = pd.read_excel(full_path, sheet_name="FIVE", header=None)
graph = df.as_matrix()

# Genetic Algorithm
print("Running Genetic Algorithm")
solution = GeneticTSP(graph, tournament_size=10, mutation_rate=0.2, num_generations=100, population_size=1000).run()

# Ant Colony Optimization
print("Running Ant Colony Optimization")
rho = 0.5
aco = AntColonyOptimization_TSP(graph=graph, t0=1, m=1000, alpha=1, beta=1,rho=0.5, delta=0.5, num_iters=20).run()

# Simulated Anneiling
print("Running Simulated Anneiling")
sa = SimulatedAnnealingTSP(graph=graph, cooling_rate=0.0003, temperature=100000).run()

# Tabu Search
print("Running Tabu Search")
TabuSearchTSP(graph = graph, iters=5, tabu_k=3).run()



#     A = np.matrix(graph)
#     G = nx.from_numpy_matrix(graph)
#     pos = nx.spring_layout(G)
#     edge_labels = dict([((u, v,), d['weight'])
#                         for u, v, d in G.edges(data=True)])
#     nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
#     nx.draw(G, pos, with_labels = True)
#     # plt.show()

