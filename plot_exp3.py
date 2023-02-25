import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--dir", default="exp3", help="Directory name for loss histories", type=str)

args = parser.parse_args()

directory = args.dir
colors = ["cyan", "black", "green", "brown", "blue", "red"]
if __name__ == "__main__":
    compressions = [f.name for f in os.scandir(directory) if f.is_dir()]
    for compression in compressions:
        Ns = [int(f.name) for f in os.scandir(directory + '/' + compression) if f.is_dir()]
        Ns.sort()
        for k, N in enumerate(Ns):
            _, _, files = next(os.walk(directory + '/' + compression + '/' + str(N)))
            iterations = len(files)
            # take the length information
            length = len(np.load(directory + '/' + compression + '/' + str(N) + '/' + files[0]))
            losses = np.zeros((iterations, length))
            for i, file in enumerate(files):
                losses[i, :] = np.load(directory + '/' + compression + '/' + str(N) + '/' + file) / N

            std_dev = np.std(losses, axis=0)
            mean_loss = np.mean(losses, axis=0)
            # lower and upper bound for %95 Confidence Interval
            lower, upper = mean_loss - 1.96 * std_dev / iterations ** 0.5, mean_loss + 1.96 * std_dev / iterations ** 0.5

            plt.fill_between(np.arange(length), lower, upper, color=colors[k], alpha=0.1)
            plt.plot(np.arange(1, length + 1), mean_loss, color=colors[k], lw=0.5)

        plt.legend(Ns, title="Agents")
        plt.title("Loss vs. time with " + compression)
        plt.xlabel("Steps")
        plt.ylabel("Loss")

        plt.savefig(directory + "/Figure-" + compression + ".png")
        plt.show()
