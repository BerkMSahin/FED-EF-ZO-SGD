import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--dir", default="exp2", help="Directory name for collision histories", type=str)

args = parser.parse_args()

directory = args.dir
colors = ["cyan", "black", "green", "brown", "blue", "red"]
if __name__ == "__main__":
    compressions = [f.name for f in os.scandir(directory) if f.is_dir()]
    for compression in compressions:
        lambdas = [float(f.name) for f in os.scandir(directory + '/' + compression) if f.is_dir()]
        lambdas.sort()
        for k, lmb in enumerate(lambdas):
            _, _, files = next(os.walk(directory + '/' + compression + '/' + str(lmb)))
            iterations = len(files)
            # take the length information
            length = len(np.load(directory + '/' + compression + '/' + str(lmb) + '/' + files[0]))
            losses = np.zeros((iterations, length))
            for i, file in enumerate(files):
                losses[i, :] = np.load(directory + '/' + compression + '/' + str(lmb) + '/' + file)

            std_dev = np.std(losses, axis=0)
            mean_loss = np.mean(losses, axis=0)
            # lower and upper bound for %95 Confidence Interval
            lower, upper = mean_loss - 1.96 * std_dev / iterations ** 0.5, mean_loss + 1.96 * std_dev / iterations ** 0.5

            plt.plot(np.arange(1, length+1), mean_loss, color=colors[k], lw=0.5)
            plt.fill_between(np.arange(length), lower, upper, color=colors[k], alpha=0.1)

        plt.legend(lambdas, title="lambdas")
        plt.title("Collisions vs. time with " + compression)
        plt.xlabel("Steps")
        plt.ylabel("Collisions")

        plt.savefig(directory + "/Figure-" + compression + ".png")
        plt.show()
