import argparse
import os

import numpy as np
import pandas as pd

from simulation import Simulation

parser = argparse.ArgumentParser()
# Arguments
parser.add_argument("--compression_name", default="qsgd", help="Compression case without error feedback", type=str)

parser.add_argument("--compression", default="False", help="Compression case without error feedback", type=str)
parser.add_argument("--error_feedback", default="False", help="Error-feedback for compressed gradients. ", type=str)
parser.add_argument("--plot_collisions", default="False", help="Switch for plotting collision vs. time", type=str)

parser.add_argument("--fraction_cord", default=.5, help="Fraction for top-k compression", type=float)
parser.add_argument("--eta", default=1, help="Learning rate for SGD", type=float)
parser.add_argument("--Lambda", default=10, help="Regularization term", type=float)
parser.add_argument("--dropout_p", default=.5, help="Dropout probability p", type=float)
parser.add_argument("--noise", default=0.5, help="Dropout probability for neighbors", type=float)

parser.add_argument("--num_bits", default=3, help="Number of bits for quantization level in qsgd", type=int)
parser.add_argument("--N", default=20, help="Number of agents", type=int)
parser.add_argument("--R", default=10, help="Radius of agent's neighbor", type=int)
parser.add_argument("--init_size", default=100, help="Initialization size", type=int)
parser.add_argument("--steps", default=1000, help="Number of steps for SGD", type=int)
parser.add_argument("--iterations", default=100, help="Number of experiments for each case", type=int)
parser.add_argument("--benchmark", default="False", help="Run benchmark test", type=str)
parser.add_argument("--animate", default="False", help="Animate", type=str)


args = parser.parse_args()


def bool_converter(value):
    if value == "False" or value == "false" or value == "FALSE":
        return False
    elif value == "True" or value == "TRUE" or value == "true":
        return True
    else :
        print("Invalid value")
        return None

if __name__ == "__main__":

    # PARAMETERS
    CNAME = args.compression_name
    ETA = args.eta  # 0.35
    STEPS = args.steps  # 10000
    ITERATIONS = args.iterations  # Number of experiments for each case
    N, R = args.N, args.R  # 20 20
    LAMBDA = args.Lambda
    INIT_SIZE = args.init_size  # 80
    ANIMATE = bool_converter(args.animate)
    FRACTION_COORDINATES = args.fraction_cord  # 0.5
    DROPOUT_P = args.dropout_p  # 0.5
    N_DROPOUT_P = args.noise  # 0.5
    PLOT_COL = bool_converter(args.plot_collisions)
    NUM_BITS = args.num_bits
    ERR_FEEDBACK = bool_converter(args.error_feedback)
    COMPRESSION = bool_converter(args.compression)
    BENCHMARK = bool_converter(args.benchmark)

    directory = "new-exps"
    if not os.path.exists(directory):
        os.makedirs(directory)


    s1 = Simulation(directory=directory, eta=ETA, steps=STEPS,
                    iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                    init_size=INIT_SIZE, animate=ANIMATE, compression=COMPRESSION,
                    fraction_coordinates=FRACTION_COORDINATES, dropout_p=DROPOUT_P,
                    n_dropout_p=N_DROPOUT_P, plot_collisions=PLOT_COL, custom_mode=True,
                    quantization_function=CNAME, num_bits=NUM_BITS, error_factor=ERR_FEEDBACK,
                    benchmark=BENCHMARK)
    s1.run()