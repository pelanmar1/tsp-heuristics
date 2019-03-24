class ABC:
    def __init__(self, graph, colony_size=30, n_iter=5000, max_trials=100):
        self.colony_size = colony_size
        self.n_iter = n_iter
        self.max_trials = max_trials
        self.graph = graph