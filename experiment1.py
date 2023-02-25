import argparse
import os

import numpy as np
import pandas as pd

from compression import compressors as c
from simulation import Simulation

parser = argparse.ArgumentParser()
# Arguments
parser.add_argument("--eta", default=0.35, help="Learning rate for SGD", type=float)
parser.add_argument("--steps", default=3000, help="Number of steps for SGD", type=int)
parser.add_argument("--iterations", default=5, help="Number of experiments for each case", type=int)
parser.add_argument("--N", default=20, help="Number of agents", type=int)
parser.add_argument("--R", default=10, help="Radius of agent's neighbor", type=int)
parser.add_argument("--Lambda", default=10, help="Regularization term", type=float)
parser.add_argument("--init_size", default=100, help="Initialization size", type=int)
parser.add_argument("--fraction_cord", default=.5, help="Fraction for top-k compression", type=float)
parser.add_argument("--dropout_p", default=.5, help="Dropout probability p", type=float)
parser.add_argument("--noise", default=.5, help="Dropout probability for neighbors", type=float)

args = parser.parse_args()

# Creates and runs a simulation instance with specified hyperparameters

if __name__ == "__main__":

    directory = "exp1"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # PARAMETERS
    ANIMATE = False
    ETA = args.eta  # 0.35
    STEPS = args.steps  # 10000
    ITERATIONS = args.iterations  # Number of experiments for each case
    N, R, LAMBDA = args.N, args.R, args.Lambda  # 20 20 5
    INIT_SIZE = args.init_size  # 80
    FRACTION_COORDINATES = args.fraction_cord  # 0.5
    DROPOUT_P = args.dropout_p  # 0.5
    N_DROPOUT_P = args.noise  # 0.5

    compression = False
    error_factor = False

    collision_table = pd.DataFrame(data=np.zeros((ITERATIONS, 11)), columns=["No comp.", c[0], c[1], c[2], c[3], c[4],
                                                                             f"{c[0]}e", f"{c[1]}e", f"{c[2]}e",
                                                                             f"{c[3]}e", f"{c[4]}e"])

    for i in range(3):

        if i == 0:
            s1 = Simulation(directory=directory, eta=ETA, steps=STEPS,
                            iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                            init_size=INIT_SIZE, animate=ANIMATE, compression=False)

            collision_hist = s1.run()
            print("Simulation without compression has been completed.")
            collision_table["No comp."] = collision_hist
        else:
            error = (i == 2)
            # TOP-K
            s1 = Simulation(directory=directory, eta=ETA, steps=STEPS,
                            iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                            init_size=INIT_SIZE, animate=ANIMATE, compression=True,
                            quantization_function=c[0], fraction_coordinates=FRACTION_COORDINATES,
                            error_factor=error, n_dropout_p=N_DROPOUT_P)

            # RAND
            s2 = Simulation(directory=directory, eta=ETA, steps=STEPS,
                            iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                            init_size=INIT_SIZE, animate=ANIMATE, compression=True,
                            quantization_function=c[1], fraction_coordinates=FRACTION_COORDINATES,
                            error_factor=error, n_dropout_p=N_DROPOUT_P)

            # DROPOUT-BIASED
            s3 = Simulation(directory=directory, eta=ETA, steps=STEPS,
                            iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                            init_size=INIT_SIZE, animate=ANIMATE, compression=True,
                            quantization_function=c[2], dropout_p=DROPOUT_P,
                            error_factor=error, n_dropout_p=N_DROPOUT_P)

            # DROPOUT-UNBIASED
            s4 = Simulation(directory=directory, eta=ETA, steps=STEPS,
                            iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                            init_size=INIT_SIZE, animate=ANIMATE, compression=True,
                            quantization_function=c[3], dropout_p=DROPOUT_P,
                            error_factor=error, n_dropout_p=N_DROPOUT_P)
            # QSGD
            s5 = Simulation(directory=directory, eta=ETA, steps=STEPS,
                            iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                            init_size=INIT_SIZE, animate=ANIMATE, compression=True,
                            quantization_function=c[4], num_bits=4,
                            error_factor=error)

            if error:
                tmp = "e"
            else:
                tmp = ""

            # Run the simulations
            collision_hist = s1.run()
            collision_table[c[0] + tmp] = collision_hist
            print("Simulation with TOP-K compression has been completed.")
            print("*" * 40)
            collision_hist = s2.run()
            collision_table[c[1] + tmp] = collision_hist
            print("Simulation with RAND compression has been completed. ")
            print("*" * 40)
            collision_hist = s3.run()
            collision_table[c[2] + tmp] = collision_hist
            print("Simulation with DROPOUT-BIASED compression has been completed. ")
            print("*" * 40)
            collision_hist = s4.run()
            collision_table[c[3] + tmp] = collision_hist
            print("Simulation with DROPOUT-UNBIASED compression has been completed. ")
            print("*" * 40)
            collision_hist = s5.run()
            collision_table[c[4] + tmp] = collision_hist
            print("Simulation with QSGD compression has been completed. ")
            print("*" * 40)

    print(collision_table)
    collision_table.to_csv(f"./{directory}/collisions.csv")  # Save the table
