
from simulation import Simulation
import numpy as np
from source import CircularSource
from agent import CircularAgent
from server import Server
from graphics import GUI
import math
import os

class CircularSimulation(Simulation):
    def __init__(self, rad=math.pi/4, radius=5, *args, **kwargs):
        super(CircularSimulation, self).__init__(*args, **kwargs)
        self.rad = rad # rotation angle in radians
        self.radius = radius # radius of circle

    def run(self):
        for i in range(self.iterations):
            np.random.seed(i) # for reproducibility
            self.collision_counter = 0
            for j in range(self.n):
                pos = np.random.uniform(low=self.init_size+100,
                                          high=3*self.init_size+100,
                                          size=(1, self.dim))
                source = CircularSource(index=j, center=pos, radius=self.radius,
                                        rad=self.rad)
                self.sources.append(source)

                agent = CircularAgent(j, self, source)
                self.agents.append(agent)

            # GET BACK HERE !!!!!!
            if len(self.sources) == 2:
                self.sources[0].center = np.array([100, 0]).reshape(1, 2)
                self.sources[1].center = np.array([-100, 0]).reshape(1, 2)
            elif len(self.sources) == 3:
                self.sources[0].center = np.array([-100, 100]).reshape(1, 2)
                self.sources[1].center = np.array([100, 100]).reshape(1, 2)
                self.sources[2].center = np.array([0, -150]).reshape(1, 2)
            elif len(self.sources) == 4:
                self.sources[0].center = np.array([-100, 100]).reshape(1, 2)
                self.sources[1].center = np.array([100, 100]).reshape(1, 2)
                self.sources[2].center = np.array([100, -100]).reshape(1, 2)
                self.sources[3].center = np.array([-100, -100]).reshape(1, 2)

            # initialize server if not testing benchmark
            if not self.benchmark:
                server = Server(self)

            for j in range(self.steps):
                if j % self.k == 0:
                    self.calculate_neighbors()
                self.losses_aggregate.append([])

                for k in range(self.n):
                    agent = self.agents[k]
                    source = self.sources[k]

                    if agent.cooldown != 0:
                        agent.cooldown -= 1

                    agent.compute_grad(source)

                self.count_collisions()
                self.collisions.append(self.collision_counter)


                for k in range(self.n):
                    agent = self.agents[k]
                    local_grad = agent.local_grad
                    if self.compressor is not None:
                        if self.error_factor:
                            local_grad_e = local_grad + agent.error
                            local_grad = self.compressor.quantize(local_grad_e.T).T
                            agent.error = local_grad_e - local_grad
                        else:
                            local_grad = self.compressor.quantize(local_grad.T).T
                    # do an update on the agent's position
                    if not self.benchmark:
                        server.local_grads.append(local_grad)
                    else:
                        agent.momentum = 0.9 * agent.momentum + self.eta * local_grad
                        agent.position = agent.position.reshape((1, 2))
                        agent.position -= agent.momentum

                if not self.benchmark:
                    server.aggregate()

                if self.animate and i + 1 == self.iterations:
                    for k in range(self.n):
                        self.agent_locs[k][j] = self.agents[k].position
                        self.source_locs[k][j] = self.sources[k].position

                path = os.getcwd()
                np.save(os.path.join(path, f"CollisionHist{i+1}"), np.array(self.collisions))

                for k in range(self.n):
                    agent = self.agents[k]
                    source = self.sources[k]
                    # teleport to the other point if agent does not come yet
                    source.move(agent)

            self.collisions = []
            self.collision_hist[i] = self.collision_counter  # save the collision count
            print(f"Experiment {i+1} has been completed.")

        if self.animate:
            gui = GUI(self.anim_width, self.anim_height, self)
            gui.animate(self.agent_locs, self.source_locs, self.sources)