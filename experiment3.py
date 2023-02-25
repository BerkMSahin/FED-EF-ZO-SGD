import argparse
import os

import numpy as np
import pandas as pd

from simulation import Simulation

parser = argparse.ArgumentParser()

parser.add_argument("--compression", default="qsgd", help="Compression case without error feedback")
parser.add_argument("--compression_e", default="qsgd", help="Compression case with error feedback")
parser.add_argument("--eta", default=0.35, help="Learning rate for SGD", type=float)
parser.add_argument("--steps", default=3000, help="Number of steps for SGD", type=int)
parser.add_argument("--iterations", default=3, help="Number of experiments for each case", type=int)
parser.add_argument("--N_list", default="5 10 15 20", help="Number of agents list", type=str)
parser.add_argument("--R", default=10, help="Radius of agent's neighbor", type=int)
parser.add_argument("--Lambda", default=10, help="Regularization term", type=str)
parser.add_argument("--fraction_cord", default=0.5, help="Fraction for top-k compression", type=float)
parser.add_argument("--dropout_p", default=0.5, help="Dropout probability p", type=float)
parser.add_argument("--noise", default=0.5, help="Noise for neighboring (between 0-1)", type=float)

args = parser.parse_args()

if __name__ == "__main__":

    # PARAMETERS
    cname = args.compression
    cname_e = args.compression_e
    ETA = args.eta  # 0.35
    STEPS = args.steps  # 10000
    ITERATIONS = args.iterations  # Number of experiments for each case
    Lambda, R = float(args.Lambda), args.R  # 5 20
    listN = [int(i) for i in args.N_list.split(" ")]
    ANIMATE = False
    FRACTION_COORDINATES = args.fraction_cord  # 0.5
    DROPOUT_P = args.dropout_p  # 0.5
    N_DROPOUT_P = args.noise  # 0.5

    compression = False
    error_factor = False

    directory = "exp3"
    if not os.path.exists(directory):
        os.makedirs(directory)

    collision_table = pd.DataFrame(data=np.zeros((ITERATIONS, 3)), columns=["No comp.", cname, cname_e + "e"])

    for exp_idx in range(len(listN)):
        # No compression case
        s1 = Simulation(directory=directory, eta=ETA, steps=STEPS,
                        iterations=ITERATIONS, n=listN[exp_idx], r=R, Lambda=Lambda,
                        init_size=5*listN[exp_idx], animate=ANIMATE, compression=False,
                        test_agents=True, n_dropout_p=N_DROPOUT_P)

        s2 = Simulation(directory=directory, eta=ETA, steps=STEPS,
                        iterations=ITERATIONS, n=listN[exp_idx], r=R, Lambda=Lambda,
                        init_size=5*listN[exp_idx], animate=ANIMATE, compression=True,
                        quantization_function=cname, error_factor=False,
                        test_agents=True, n_dropout_p=N_DROPOUT_P)

        s3 = Simulation(directory=directory, eta=ETA, steps=STEPS,
                        iterations=ITERATIONS, n=listN[exp_idx], r=R, Lambda=Lambda,
                        init_size=5*listN[exp_idx], animate=ANIMATE, compression=True,
                        quantization_function=cname_e, error_factor=True,
                        test_agents=True, n_dropout_p=N_DROPOUT_P)

        collision_hist = s1.run()
        collision_table["No comp."] = collision_hist
        print(f"Simulation with no compression has been completed for N={listN[exp_idx]}")
        print("*" * 40)
        collision_hist = s2.run()
        collision_table[cname] = collision_hist
        print(f"Simulation with {cname} compression has been completed for N={listN[exp_idx]} ")
        print("*" * 40)
        collision_hist = s3.run()
        collision_table[cname + "E"] = collision_hist
        print(f"Simulation with {cname_e} compression + error feedback has been completed for N={listN[exp_idx]} ")
        print("*" * 40)

    print(collision_table)
    collision_table.to_csv(f"./{directory}/collisions.csv")  # Save the table
