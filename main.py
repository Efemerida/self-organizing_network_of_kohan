import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

import utils
from Node import Node 

from Kohan import Kohan_Network_Map
from Kohan_2 import Kohan_Network_Line

from Node import Node

def normalization(data):
    normalisation = []
    for i in range(len(data)):
        normalisation.append([])
        sum_pow = 0
        for j in range(len(data[0])):
            sum_pow += data[i][j] ** 2
        sqrt = np.sqrt(sum_pow)
        for j in range(len(data[0])):
            normalisation[i].append(data[i][j] / sqrt)
    return normalisation

def show_dataset(dataset):
    arr = []
    p = 0
    for i in range(50):
        arr.append([])
        for j in range(3):
            arr[i].append(np.sum(dataset[p]))
            p += 1
    plt.imshow(arr, cmap='brg')
    plt.show()

def show_kohan_map(network):
    arr = []

    sum = 0
    z = 0
    p = 0
    for i in range(10):
        arr.append([])
        for j in range(10):
            arr[i].append(network.nodes[i][j].color)

    plt.close()
    plt.imshow(arr, cmap='brg')
    plt.show()


def test_kohan_map():
    iris_dataset = datasets.load_iris()["data"]
    fruit_dataset = utils.get_fruit_dataset()
    wine_dataset = utils.get_wine_dataset()

    show_dataset(wine_dataset)

    normalization_dataset = normalization(wine_dataset)
    normalization_dataset = np.random.permutation(normalization_dataset)

    kn2 = Kohan_Network_Map(10, 10, normalization_dataset)
    kn2.init_weights()
    kn2.update_color()

    show_kohan_map(kn2)
    
    kn2.learn(normalization_dataset)
    kn2.update_color()

    show_kohan_map(kn2)


def test_kohan_line():
    iris_dataset = datasets.load_iris()["data"]
    fruit_dataset = utils.get_fruit_dataset()
    wine_dataset = utils.get_wine_dataset()

    show_dataset(iris_dataset)

    normalization_dataset = normalization(iris_dataset)
    normalization_dataset = np.random.permutation(normalization_dataset)

    kn = Kohan_Network_Line(50, 3, normalization_dataset)
    kn.init_weights()
    kn.update_color()

    arr = []

    sum = 0
    z = 0
    p = 0
    for i in range(50):
        arr.append([])
        for j in range(3):
            arr[i].append(kn.nodes[p].color)
            p += 1

    print(arr)

    kn.learn(normalization_dataset)

    arr = []
    sum = 0
    z = 0
    p = 0
    for i in range(50):
        arr.append([])
        for j in range(3):
            arr[i].append(kn.nodes[p].color)
            p += 1

    print(arr)



    plt.close()
    plt.imshow(arr, cmap='brg')
    plt.show()





def main():
    test_kohan_line()


if __name__=='__main__':
    main()