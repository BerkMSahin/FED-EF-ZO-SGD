import argparse

import matplotlib.pyplot as plt
import numpy as np
import os

parser = argparse.ArgumentParser()

parser.add_argument("--dir", default="exp1", help="Directory name for loss histories")

args = parser.parse_args()

directory = args.dir
colors = ["red", "black", "green", "brown", "blue", "yellow"]
if __name__ == "__main__":
    # Plot the compressions without error feedback
    compressions = [f.name for f in os.scandir(directory + "/no-error") if f.is_dir()]
    for k, compression in enumerate(compressions):
        _, _, files = next(os.walk(directory + "/no-error/" + compression))
        iterations = len(files)
        # take the length information
        length = len(np.load(directory + '/no-error/' + compression + '/' + files[0]))
        losses = np.zeros((iterations, length))
        for i, file in enumerate(files):
            losses[i,:] = np.load(directory + '/no-error/' + compression + '/' + file)

        std_dev = np.std(losses, axis=0)
        mean_loss = np.mean(losses, axis=0)
        # lower and upper bound for %95 Confidence Interval
        lower, upper = mean_loss - 1.96 * std_dev / iterations**0.5, mean_loss + 1.96 * std_dev / iterations**0.5

        plt.title("Loss vs. time without error factor")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.plot(np.arange(length), mean_loss, color=colors[k], lw=0.5)
        plt.fill_between(np.arange(length), lower, upper, color=colors[k], alpha=0.1)

    plt.legend(compressions)
    plt.savefig(directory + "/no-error/Figure.png")
    plt.close()

    # Plot the compressions with error feedback
    compressions = [f.name for f in os.scandir(directory + "/error") if f.is_dir()]
    compressions.remove("dropout-unbiased")
    for k, compression in enumerate(compressions):
        _, _, files = next(os.walk(directory + "/error/" + compression))
        iterations = len(files)
        # take the length information
        length = len(np.load(directory + '/error/' + compression + '/' + files[0]))
        losses = np.zeros((iterations, length))
        for file in files:
            losses[i,:] = np.load(directory + '/error/' + compression + '/' + file)

        std_dev = np.std(losses, axis=0)
        mean_loss = np.mean(losses, axis=0)
        # lower and upper bound for %95 Confidence Interval
        lower, upper = mean_loss - 1.96 * std_dev / iterations ** 0.5, mean_loss + 1.96 * std_dev / iterations ** 0.5

        plt.title("Loss vs. time with error factor")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.plot(np.arange(length), mean_loss, color=colors[k], lw=0.5)
        plt.legend(compressions)
        plt.fill_between(np.arange(length), lower, upper, color=colors[k], alpha=0.1)

    plt.savefig(directory + "/error/Figure.png")
    plt.close()