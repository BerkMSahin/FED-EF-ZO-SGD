import numpy as np
from numpy.linalg import norm

class Server:
    def __init__(self, simulation):
        self.simulation = simulation
        self.local_grads = []
        self.grad_X = np.zeros((simulation.n, simulation.dim))
        self.X = np.zeros((simulation.n, simulation.dim))
        for i in range(simulation.n):
            agent = simulation.agents[i]
            self.X[i, :] = agent.position

    # Aggregates the local gradients of each agent and determines
    # their subsequent positions using the aggregated gradient
    def aggregate(self):
        self.grad_X = np.mean(np.array(self.local_grads), axis=1)
        row_norms = np.sqrt((self.grad_X**2).sum(axis=1, keepdims=True))
        self.grad_X = self.grad_X / (row_norms + 10e-9)
        self.X -= self.simulation.eta * self.grad_X

        for i in range(self.simulation.n):
            self.simulation.agents[i].position = self.X[i, :]
            self.simulation.agents[i].velocity = -self.simulation.eta * self.grad_X[i, :]

        self.local_grads = []
