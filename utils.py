import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_graph(data, labels, legends, title):
    """
    Plot multiple graphs in same plot

    :param data: data of the graphs to be plotted
    :param labels: x- and y-label
    :param legends: legends for the graphs
    :param title: Title of the graph
    """
    x = np.arange(1, len(data[0]) + 1)
    for to_plot in data:
        plt.plot(x, to_plot)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legends)
    plt.show()
    plt.savefig('{}.png'.format(title))


def plot_training_data(data):
    """
    Plot training data, loss over epochs.
    """
    training_data = np.array(data).T
    plot_graph(training_data, ["Epoch", "Cross-entropy-loss"], ["Training loss", "Validation loss"],
               "Loss over epochs")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
