import numpy as np


def load_dataset():
    with np.load("mnist.npz") as f:
        # convert from RGB to Unit RGB
        x_train = f['x_train'].astype("float32") / 255

        # reshape from (60000, 28, 28) into (60000, 784)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

        # labels
        y_train = f['y_train']

        # convert to output layer format
        y_train = np.eye(10)[y_train]

        return x_train, y_train


def load_test_dataset():
    with np.load("mnist.npz") as f:
        # convert from RGB to Unit RGB
        x_test = f['x_test'].astype("float32") / 255

        # reshape from (60000, 28, 28) into (60000, 784)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

        # labels
        y_test = f['y_test']

        # convert to output layer format
        y_test = np.eye(10)[y_test]

        return x_test, y_test


def get_fruit_dataset():
    fruit_dataset_file = np.genfromtxt('fruit.csv', delimiter=';')
    fruit_dataset = []

    for i in range(0, len(fruit_dataset_file) - 3):
        fruit_dataset.append([])
        for j in range(len(fruit_dataset_file[1]) - 1):
            fruit_dataset[i].append(fruit_dataset_file[i + 1][j])

    return fruit_dataset


def get_wine_dataset():
    fruit_wine_file = np.genfromtxt('wine.csv', delimiter=',')
    wine_dataset = []

    for i in range(0, len(fruit_wine_file) - 1):
        wine_dataset.append([])
        for j in range(len(fruit_wine_file[1])-1):
            wine_dataset[i].append(fruit_wine_file[i + 1][j])

    return wine_dataset
