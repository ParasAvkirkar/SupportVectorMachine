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
    # TODO: remove this logic
    indices_to_remove = set([i for i in range(X.shape[0])])

    m = X.shape[0]
    feature_len = X.shape[1]

    max_fval = get_max_f_value(X, y)
    step_size = 0.001 * max_fval

    # T = iterations
    T = 50000
    lambda_param = 0.01

    w_t = np.zeros(feature_len)
    weight_sum = np.zeros(feature_len)
    for t in range(1, T + 1):
        # if len(indices_to_remove) == 0:
        #     break
        i = np.random.choice(m)  # gets a uniformly random integer index between [0, m-1]

        y_i = y[i, 0]
        x_i = X[i]

        loss_differentiation = np.zeros(feature_len)
        reg_differentiation = lambda_param * w_t

        # setting bias regularization to 0
        reg_differentiation[0] = 0
        if y_i * np.dot(w_t, x_i) < 1:
            loss_differentiation = np.add(reg_differentiation, (-1 * y_i) * x_i)
        else:
            loss_differentiation = reg_differentiation

        w_t = w_t - step_size * loss_differentiation

        # keeping a limit on step_size of 0.0001
        if step_size * 0.1 >= 0.0001:
            step_size = 0.1 * step_size

        weight_sum = np.add(w_t, weight_sum)
        if t % 10 == 0:
            best_current_weight = (1.0/t) * weight_sum
            error = 1 - test(X, y, w_t)
            # print("Iteration number: " + str(t) + " Training error: " + str(error) + " Not removed: " + str(len(indices_to_remove)))

        indices_to_remove.discard(i)


    w = (1.0 / T) * weight_sum
    # print("Indices not removed: " + str(len(indices_to_remove)))

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

    positive_X = data_dict[1]
    negative_X = data_dict[-1]

    fig, ax = plt.subplots()
    ax.scatter(positive_X[:, 1], positive_X[:, 2], c='g')
    ax.scatter(negative_X[:, 1], negative_X[:, 2], c='r')

    ax.legend()
    ax.grid(True)

    min_x, max_x = plt.xlim()
    min_x = min_x - 1
    max_x = max_x + 1
    # Taking a set of points to use to plot a line across a grid
    x_axis = np.linspace(min_x, max_x, 10000)

    # central hyperplane
    y_axis = (-hyper_plane[1]/hyper_plane[2]) * x_axis - hyper_plane[0]/hyper_plane[2]
    ax.plot(x_axis, y_axis, '-b')

    # positive hyperplane
    y_axis = (-hyper_plane[1] / hyper_plane[2]) * x_axis - hyper_plane[0] / hyper_plane[2] + (1.0/hyper_plane[2])
    ax.plot(x_axis, y_axis, '--g')

    # negative hyperplane
    y_axis = (-hyper_plane[1] / hyper_plane[2]) * x_axis - hyper_plane[0] / hyper_plane[2] - (1.0/hyper_plane[2])
    ax.plot(x_axis, y_axis, '--r')

    plt.show()

def main():
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
    # print("Norm: " + str(np.linalg.norm(w)))
    accuracy = test(test_x, test_y, w)

    print("Validation Accuracy: " + str(accuracy * 100) + "%")

    whole_accuracy = test(X1, y, w)
    print("Whole accuracy: " + str(whole_accuracy * 100) + "%")

    draw(X1, y, hyper_plane=w)

    # TODO: REMOVE
    return whole_accuracy


if __name__ == "__main__":
    found = False
    for i in range(1):
        acc = main()
        if acc < 1.0:
            found = True
            break

    if found:
        print("Whole accuracy was less than 100.0")
    else:
        print("Keep it up!")
