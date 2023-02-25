import numpy as np
from numpy.linalg import norm


class Agent:
    def __init__(self, index, simulation):
        self.simulation = simulation
        self.position = np.random.uniform(low=-simulation.init_size,
                                          high=simulation.init_size,
                                          size=(1, simulation.dim))
        self.index = index
        self.velocity = np.zeros((1, simulation.dim))
        self.momentum = 0
        self.local_grad = np.zeros((simulation.n, simulation.dim))
        self.error = np.zeros((simulation.n, simulation.dim))
        self.cooldown = 0
        self.mu = .3
        #print(f"agent {index} initial position: {self.position}")
    # Loss functions for gradient estimation, with respect to the pursued source
    def loss(self, source):
        return .5 * (norm(self.position - source.position) ** 2)

    def loss_plus(self, source, u):
        return .5 * (norm(self.position + self.mu * u - (source.position + .5 * source.velocity)) ** 2)

    # Loss functions for gradient estimation, with respect to neighboring agents
    # (used for the regularization terms)
    def loss_reg(self, neighbor):
        return self.simulation.Lambda * (norm(self.position - neighbor.position)**2 - self.simulation.r**2)

    def loss_reg_plus(self, neighbor, u):
        return self.simulation.Lambda * (norm(self.position +
                                              self.mu * u -
                                              (neighbor.position + .5 * neighbor.velocity))**2 -
                                         self.simulation.r**2)

    # Computes the local gradient of the agent
    def compute_grad(self, source):
        u = np.random.standard_normal((1, self.simulation.dim))
        grad_i = u * (self.loss_plus(source, u) - self.loss(source)) / self.mu
        if not self.simulation.benchmark:
            local_grad = np.zeros((self.simulation.n, self.simulation.dim))
            local_grad[self.index, :] = grad_i

            for neighbor in self.simulation.detected_neighbors[self.index]:
                u = np.random.standard_normal((1, self.simulation.dim))
                grad_j = u * (self.loss_reg_plus(neighbor, u) - self.loss_reg(neighbor)) / self.mu
                local_grad[neighbor.index, :] = -grad_j
        else:
            local_grad = grad_i / norm(grad_i)

        self.local_grad = local_grad

class CircularAgent(Agent):
    def __init__(self, index, simulation, source):
        super(CircularAgent, self).__init__(index, simulation)
        rotate = source.rotate
        # initialize agent from 1 rad movement behind
        rotate[0, 1] *= -1
        rotate[1, 0] *= -1
        centered_pos = source.position - source.center
        self.position = np.matmul(rotate, centered_pos.T).T + source.center

    # velocity was removed
    def loss_plus(self, source, u):
        return .5 * (norm(self.position + self.mu * u - source.position) ** 2)









