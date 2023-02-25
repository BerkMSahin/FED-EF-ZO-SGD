import numpy as np
from numpy.linalg import norm


class Source:
    def __init__(self, index, beta, simulation):
        self.simulation = simulation
        self.position = np.random.uniform(low=simulation.init_size+100,
                                          high=3*simulation.init_size+100,
                                          size=(1, simulation.dim))
        self.index = index
        self.beta = beta
        self.velocity = 0

    # Determines velocity in order to actively avoid the chasing agent
    def set_velocity(self, agent):
        difference = self.position - agent.position
        self.velocity = self.beta * difference / norm(difference)

    def move(self):
        self.position += self.velocity

# sources teleporting around circle
class CircularSource:
    def __init__(self, index, center, radius, rad, threshold=3):
        self.radius = radius
        self.center = center
        self.index = index
        self.threshold = threshold
        # init position randomly on circle
        pos = np.random.normal(loc=0, scale=1, size=(1,2))
        # random position on circle arc
        self.position = (pos / np.linalg.norm(pos)) * radius
        # create rotation matrix
        c, s = np.cos(rad), np.sin(rad)
        self.rotate = np.array([[c, -s], [s, c]])

    def move(self, agent):
        if np.linalg.norm(self.position - agent.position, 2) < self.threshold:
            centered_pos = self.position - self.center
            self.position = np.matmul(self.rotate, centered_pos.T).T + self.center






