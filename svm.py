import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


# returns two numpy arrays containing positive and negative samples
def split_data_by_labels(X, y):
    m = X.shape[0]
    positive_X = []
    negative_X = []
    for i in range(m):
        if y[i] <= 0:
            negative_X.append([[X[i, 0], X[i, 1]]])
        else:
            positive_X.append([[X[i, 0], X[i, 1]]])

    len_p = len(positive_X)
    len_n = len(negative_X)
    positive_X = np.array(positive_X).reshape(len_p, 2)
    negative_X = np.array(negative_X).reshape(len_n, 2)

    return positive_X, negative_X


def get_max_f_value(X, y):
    positive_X, negative_X = split_data_by_labels(X, y)
    data_dict = {-1: negative_X, 1: positive_X}

    max_fval = float('-inf')
    for y_i in data_dict:
        if np.amax(data_dict[y_i]) > max_fval:
            max_fval = np.amax(data_dict[y_i])

    return max_fval


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
# return weight_vector
def train(X, y, iterations=100, lambda_param=10):
    m = X.shape[0]
    feature_len = X.shape[1]

    max_fval = get_max_f_value(X, y)
    step_size = 0.1 * max_fval

    T = iterations
    theta_t = np.zeros(feature_len)
    all_weights = []
    for t in range(1, T + 1):
        w_t = (1.0 / lambda_param) * step_size * theta_t
        i = np.random.choice(m)  # gets a uniformly random integer index between [0, m-1]

        y_i = y[i, 0]
        x_i = X[i]

        if y_i * np.dot(w_t, x_i) < 1:
            theta_t = theta_t + y_i * x_i

        all_weights.append(w_t)

        step_size = 0.1 * step_size

    all_weights = np.array(all_weights)
    w = (1.0 / T) * np.sum(all_weights, axis=0)

    return w


def test(X, y, w):
    test_sample_len = X.shape[0]

    mistakes = 0.0

    for i in range(test_sample_len):
        x_i = X[i]
        y_i = y[i, 0]

        prediction = predict(w, x_i)
        if prediction != y_i:
            mistakes += 1.0

    accuracy = 1 - (mistakes/test_sample_len)
    print("Accuracy: " + str(accuracy * 100) + "%")


def predict(w, x):
    return 1.0 if np.dot(w, x) >= 0 else -1.0


if __name__ == "__main__":
    X0, y = make_blobs(n_samples=100, n_features=2, centers=2,
                       cluster_std=1.05, random_state=10)

    X1 = np.c_[np.ones((X0.shape[0])), X0]  # add one to the x-values to incorporate bias

    # plot(X0, y)

    y = np.where(y == 0, -1.0, y)
    y = y.reshape(y.shape[0], 1)

    m = X1.shape[0]
    split_point = int(math.ceil(0.8 * m))

    train_x, train_y = X1[:split_point, :], y[:split_point, :]
    test_x, test_y = X1[split_point:, :], y[split_point:, :]

    w = train(train_x, train_y)
    test(test_x, test_y, w)

