# TODO: Remove use of tqdm
from tqdm.auto import tqdm

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


# returns two numpy arrays containing positive and negative samples
def split_data_by_labels(X, y):
    m = X.shape[0]
    feature_len = X.shape[1]

    positive_X = []
    negative_X = []
    for i in range(m):
        if y[i] <= 0:
            negative_X.append(X[i])
        else:
            positive_X.append(X[i])

    len_p = len(positive_X)
    len_n = len(negative_X)
    positive_X = np.array(positive_X).reshape(len_p, feature_len)
    negative_X = np.array(negative_X).reshape(len_n, feature_len)

    data_dict = {-1: negative_X, 1: positive_X}

    return data_dict


def get_max_f_value(X, y):
    data_dict = split_data_by_labels(X, y)

    max_fval = float('-inf')
    for y_i in data_dict:
        if np.amax(data_dict[y_i]) > max_fval:
            max_fval = np.amax(data_dict[y_i])

    return max_fval

# TODO: remove this function later (using draw function instead)
def plot(X, y):
    data_dict = split_data_by_labels(X, y)

    positive_X = data_dict[-1]
    negative_X = data_dict[1]

    fig, ax = plt.subplots()
    ax.scatter(positive_X[:, 0], positive_X[:, 1], c='r')
    ax.scatter(negative_X[:, 0], negative_X[:, 1], c='b')

    ax.legend()
    ax.grid(True)

    plt.show()


# soft-svm with hinge-loss with gradient descent
# because hard-svm is deterministic and we are asked to implement gradient descent
# return weight_vector
def train(X, y):
    m = X.shape[0]
    feature_len = X.shape[1]

    max_fval = get_max_f_value(X, y)
    step_size = 0.1 * max_fval

    # T = iterations
    T = 3000
    lambda_param = 1

    theta_t = np.zeros(feature_len)
    weight_sum = np.zeros(feature_len)
    for t in range(1, T + 1):
        w_t = (1.0 / lambda_param) * step_size * theta_t

        weight_sum = np.add(w_t, weight_sum)
        current_weight = (1.0 / t) * weight_sum

        if t % 100 == 0:
            print("Iteration number: " + str(t) + " Error: " + str(1 - test(X, y, current_weight)))

        i = np.random.choice(m)  # gets a uniformly random integer index between [0, m-1]

        y_i = y[i, 0]
        x_i = X[i]

        if y_i * np.dot(w_t, x_i) < 1:
            theta_t = theta_t + y_i * x_i
        else:
            # don't need to do anything
            pass

        step_size = 0.1 * step_size

    w = (1.0 / T) * np.array(weight_sum)

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
    return accuracy


def predict(w, x):
    return 1.0 if np.dot(w, x) >= 0 else -1.0

def draw(X, y, hyper_plane):
    data_dict = split_data_by_labels(X, y)

    positive_X = data_dict[-1]
    negative_X = data_dict[1]

    fig, ax = plt.subplots()
    ax.scatter(positive_X[:, 1], positive_X[:, 2], c='r')
    ax.scatter(negative_X[:, 1], negative_X[:, 2], c='b')

    ax.legend()
    ax.grid(True)

    min_v = np.amin(X) - 1
    max_v = np.amax(X) + 1

    x_axis = np.linspace(min_v, max_v, int((max_v - min_v)/0.01))
    y_axis = (-hyper_plane[1]/hyper_plane[2]) * x_axis - hyper_plane[0]/hyper_plane[2]

    ax.plot(x_axis, y_axis, '-g')


    plt.show()

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

    data_dict = split_data_by_labels(train_x, train_y)
    w = train(train_x, train_y)
    print(str(w))
    print("Norm: " + str(np.linalg.norm(w)))
    accuracy = test(test_x, test_y, w)

    print("Validation Accuracy: " + str(accuracy * 100) + "%")
    print("Whole accuracy: " + str(test(X1, y, w) * 100) + "%")

    draw(X1, y, hyper_plane=w)
