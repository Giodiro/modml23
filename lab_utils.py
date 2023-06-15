import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


__all__ = [
    "create_random_data",
    "plot_dataset",
]


def plot_dataset(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[y == -1][:,0], X[y == -1][:,1], alpha=0.5)
    ax.scatter(X[y == 1][:,0], X[y == 1][:,1], alpha=0.5)


def _gen_linear_data(n_samples, noise_level):
    fst_half = n_samples // 2
    snd_half = n_samples - fst_half

    Y = np.ones((n_samples, ))
    Y[:fst_half] = -1

    X1 = np.random.normal([5, 5], scale=[1*noise_level], size=(fst_half, 2))
    X2 = np.random.normal([8, 5], scale=[1*noise_level], size=(snd_half, 2))

    return np.concatenate((X1, X2), 0), Y

def _gen_moons(n_samples, noise_level):
    X, Y = datasets.make_moons(n_samples=n_samples, shuffle=False, noise=noise_level)
    Y[Y == 0] = -1

    return X, Y

def _gen_circles(n_samples, noise_level):
    X, Y = datasets.make_circles(n_samples=n_samples, shuffle=True, noise=noise_level)
    Y[Y == 0] = -1
    return X, Y

def create_random_data(n_samples, noise_level, dataset="linear", seed=0):
    """Generates a random dataset. Can generate 'linear', 'moons' or 'circles'.

    Parameters
    ----------
    n_samples
        The total number of samples. These will be equally divided between positive
        and negative samples.
    noise_level
        The amount of noise: higher noise -> harder problem. The meaning of the noise
        is different for each dataset.
    dataset
        A string to specify the desired dataset. Can be 'linear', 'moons', 'circles'.
    seed
        Random seed for reproducibility.

    Returns
    -------
    X
        A 2D array of features
    Y
        A vector of targets (-1 or 1)
    """
    np.random.seed(seed)

    if dataset.lower() == "linear":
        return _gen_linear_data(n_samples, noise_level)
    elif dataset.lower() == "moons":
        return _gen_moons(n_samples, noise_level)
    elif dataset.lower() == "circles":
        return _gen_circles(n_samples, noise_level)
    else:
        raise ValueError(("Dataset '%s' is not valid. Valid datasets are:"
                          " 'linear', 'moons', 'circles'") % (dataset))
