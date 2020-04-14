import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

def split_data_by_labels(X, y):
    m = X.shape[0]
    positive_X = []
    negative_X = []
    for i in range(m):
        if y[i] == 0:
            negative_X.append([[X[i, 0], X[i, 1]]])
        else:
            positive_X.append([[X[i, 0], X[i, 1]]])

    len_p = len(positive_X)
    len_n = len(negative_X)
    positive_X = np.array(positive_X).reshape(len_p, 2)
    negative_X = np.array(negative_X).reshape(len_n, 2)

    return positive_X, negative_X


def plot(X, y):
    positive_X, negative_X = split_data_by_labels(X, y)

    fig, ax = plt.subplots()
    ax.scatter(positive_X[:, 0], positive_X[:, 1], c='r')
    ax.scatter(negative_X[:, 0], negative_X[:, 1], c='b')

    ax.legend()
    ax.grid(True)

    plt.show()

# soft-svm with hinge-loss with gradient descent
# because hard-svm is deterministic and we are asked to implement gradient descent
def train(X, y):
    pass


if __name__ == "__main__":
    X0, y = make_blobs(n_samples=100, n_features = 2, centers=2,
    cluster_std=1.05, random_state=10)

    X1 = np.c_[np.ones((X0.shape[0])), X0] # add one to the x-values to incorporate bias

    plot(X0, y)