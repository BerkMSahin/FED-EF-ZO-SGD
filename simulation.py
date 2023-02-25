import numpy as np
from numpy.linalg import norm
from server import Server
from agent import Agent
from source import Source
from graphics import GUI
from compression import Compression
import matplotlib.pyplot as plt
import os


class Simulation:
    def __init__(self, directory, eta=100, beta=.1, dim=2,
                 steps=10000, iterations=10, n=5, r=5,
                 Lambda=.1, init_size=100, k=1, animate=False,
                 anim_width=1600, anim_height=900, compression=False,
                 num_bits=3, quantization_function="top", dropout_p=0.5,
                 fraction_coordinates=0.5, error_factor=False, plot=False,
                 n_dropout=True, n_dropout_p=0.5, cooldown=3,
                 test_lambda=False, test_agents=False, plot_collisions=False,
                 custom_mode=False, benchmark=False):
        self.eta = eta
        self.beta = beta
        self.dim = dim
        self.steps = steps
        self.iterations = iterations
        self.n = n
        self.r = r
        self.Lambda = Lambda
        self.init_size = init_size
        self.k = k
        self.animate = animate
        self.anim_width = anim_width
        self.anim_height = anim_height
        self.losses_aggregate = []
        self.collisions = []
        self.global_losses = []
        self.agents = []
        self.agent_locs = np.zeros((n, steps, dim))
        self.sources = []
        self.source_locs = np.zeros((n, steps, dim))
        self.neighbors_aggregate = {}
        self.detected_neighbors = {}
        # Compression parameters
        self.compression = compression
        self.quantization_function = quantization_function
        self.compressor = Compression(num_bits, quantization_function, dropout_p,
                                      fraction_coordinates) if compression else None
        self.error_factor = error_factor
        self.collision_counter = 0  # counts the collisions between agents
        self.plot = plot
        self.collision_hist = np.zeros((self.iterations, 1))
        self.n_dropout = n_dropout
        self.n_dropout_p = n_dropout_p
        self.directory = directory
        self.cooldown = cooldown
        self.test_lambda = test_lambda
        self.test_agents = test_agents
        self.path = ""
        self.plot_collisions = plot_collisions
        self.custom_mode = custom_mode
        self.benchmark = benchmark

    def loss(self, agent, source):
        #index = agent.index
        loss = agent.loss(source)
        #for neighbor in self.neighbors_aggregate[index]:
        #    loss -= agent.loss_reg(neighbor)
        return loss

    # Calculates and updates each agent's list of neighboring agents
    def calculate_neighbors(self):
        x1 = 1
        x2 = 1
        self.neighbors_aggregate.clear()
        self.detected_neighbors.clear()

        for i in range(self.n):
            self.neighbors_aggregate[i] = []
            self.detected_neighbors[i] = []

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if norm(self.agents[i].position - self.agents[j].position) < self.r:
                    self.neighbors_aggregate[i].append(self.agents[j])
                    self.neighbors_aggregate[j].append(self.agents[i])
                    if self.n_dropout:
                        x1 = np.random.rand()
                        x2 = np.random.rand()
                    if x1 > self.n_dropout_p:
                        self.detected_neighbors[i].append(self.agents[j])
                    if x2 > self.n_dropout_p:
                        self.detected_neighbors[j].append(self.agents[i])

    def count_collisions(self):
        for agent_idx in range(self.n):
            agent = self.agents[agent_idx]
            for neighbor in self.neighbors_aggregate[agent_idx]:
                # To make the dimensions consistent
                agent.position = agent.position.reshape(2, 1)
                neighbor.position = neighbor.position.reshape(2, 1)

                if np.linalg.norm(agent.position - neighbor.position) <= 3 \
                        and agent.cooldown == 0 \
                        and neighbor.cooldown == 0:
                    self.collision_counter += 1
                    agent.cooldown = self.cooldown
                    neighbor.cooldown = self.cooldown

    def run(self):
        exp_no = len(os.listdir(self.directory)) // 2
        for i in range(self.iterations):
            np.random.seed(i+5)  # set random seed
            self.collision_counter = 0
            for j in range(self.n):
                agent = Agent(j, self)
                self.agents.append(agent)

                source = Source(j, self.beta, self)
                self.sources.append(source)

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

                for k in range(self.n):
                    agent = self.agents[k]
                    source = self.sources[k]

                    source.set_velocity(agent)
                    source.move()

                    error = self.loss(agent, source)
                    self.losses_aggregate[j].append(error)

                self.losses_aggregate[j] = np.array(self.losses_aggregate[j])

            global_loss = np.mean(np.array(self.losses_aggregate), axis=1)
            self.agents = []
            self.sources = []
            self.losses_aggregate = []
            self.global_losses.append(global_loss)

            if not self.custom_mode:
                # Save the losses and collisions
                if self.test_agents or self.test_lambda:
                    if self.compression:
                        e = '-e' if self.error_factor else ''
                        self.path = self.directory + '/' + self.quantization_function + e
                    else:
                        self.path = self.directory + '/' + "no-comp"

                    if not os.path.exists(self.path):
                        os.makedirs(self.path)

                    if self.test_lambda:
                        path = self.path + '/' + str(self.Lambda)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        if self.plot_collisions:
                            np.save(path + '/' + str(i), np.array(self.collisions))
                        else:
                            np.save(path + '/' + str(i), global_loss)
                    elif self.test_agents:
                        path = self.path + '/' + str(self.n)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        np.save(path + '/' + str(i), global_loss)
                else:
                    path_error = self.directory + '/error'
                    path_no_error = self.directory + '/no-error'

                    if not os.path.exists(path_error):
                        os.makedirs(path_error)

                    if not os.path.exists(path_no_error):
                        os.makedirs(path_no_error)

                    if self.compression:
                        path = path_error + '/' + self.quantization_function if self.error_factor \
                            else path_no_error + '/' + self.quantization_function
                        if not os.path.exists(path):
                            os.makedirs(path)
                        path += '/' + str(i)
                        np.save(path, global_loss)
                    else:
                        path = path_error + '/no-comp'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        np.save(path + '/' + str(i), global_loss)

                        path = path_no_error + '/no-comp'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        np.save(path + '/' + str(i), global_loss)

            else:
                path = self.directory + '/' + str(exp_no)
                # Check the compression
                if self.compression:
                    compressor_name = self.compressor.quantization_function
                    path += 'C' + compressor_name
                    if compressor_name == "qsgd":
                        path += 'Bits' + str(self.compressor.num_bits)
                    elif compressor_name == "top" or compressor_name == "rand":
                        path += 'k' + str(self.compressor.fraction_coordinates)
                    elif compressor_name == "dropout-unbiased" or compressor_name == "dropout-biased":
                        path += 'p' + str(self.compressor.dropout_p)
                    else:
                        print("Invalid compressor.")
                else:
                    path += "NoC"

                # Check the error feedback
                if self.error_factor:
                    path += "ErrY"
                else:
                    path += "ErrN"

                path += 'N' + str(self.n) + 'R'+ str(self.r) + 'Lmb' + str(self.Lambda) + 'NDrop' + str(self.n_dropout_p) + "Eta" + str(self.eta)

                # Save the collisions and losses
                if not os.path.exists(path):
                    os.makedirs(path)

                if not os.path.exists(path+'Coll'):
                    os.makedirs(path+'Coll')

                np.save(path + 'Coll/' + str(i), np.array(self.collisions))
                np.save(path + '/' + str(i), global_loss)

            self.collisions = []
            self.collision_hist[i] = self.collision_counter  # save the collision count
            print(f"Experiment {i} has been completed.")

        final_loss = np.mean(np.array(self.global_losses), axis=0)
        print("Final loss:", final_loss[-1])
        print(f"Number of collisions: {self.collision_counter}")

        if self.animate:
            gui = GUI(self.anim_width, self.anim_height, self)
            gui.animate(self.agent_locs, self.source_locs)

            np.save(f"./{self.directory}/loss_hist.npy", final_loss)

        if self.plot:
            plt.xlabel('Steps')
            plt.ylabel('Loss value')
            plt.title('Zeroth order federated tracking, n=' + str(self.n) +
                      ", r=" + str(self.r) +
                      ", lambda=" + str(self.Lambda) + ",\n" +
                      "iterations=" + str(self.iterations) +
                      ", final loss=" + str(final_loss[-1]) +
                      ", Compression: " + self.quantization_function)
            plt.plot(final_loss)
            plt.grid(which="major")
            plt.show()
        return self.collision_hist
