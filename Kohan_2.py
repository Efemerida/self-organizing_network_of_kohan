import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

import utils
from Node import Node

class Kohan_Network_Line:
    nodes = []

    alfa = 0.1
    epochs = 4
    p_min = 0.5
    T = 15
    L0 = 0.3
    lamda = 0

    frames = []

    excima0 = 0
    potential = []

    def __init__(self, N, M, input_data):
        self.M = M
        self.N = N
        self.input_data = input_data
        self.potential = [1 / (N * M) for _ in range(N * M)]
        self.excima0 = max(N, M) / 2
        self.lamda = self.T / np.log(self.excima0)

    def init_weights(self):
        for i in range(self.N * self.M):
            weights = [0] * len(self.input_data[0])
            tt = np.random.randint(3, 80, (1, 4))
            for p in range(4):
                weights[p] = tt[0][p] / 100
            self.nodes.append(Node(i, 0, weights))

    def calc_cluster(self, input_data):
        d = np.zeros((self.N * self.M))
        for i in range(self.N * self.M):
            if self.potential[i] < self.p_min: continue
            sum = 0
            for z in range(len(input_data)):
                sum += ((input_data[z] - self.nodes[i].weights[z]) ** 2)
            d[i] = np.sqrt(sum)

        return np.argmin(d)

    def get_neighbours(self, node, t):
        neighbours = []
        for i in range(self.N * self.M):
            sum = 0
            for z in range(len(node.weights)):
                sum += (node.weights[z] - self.nodes[i].weights[z]) ** 2
            d = np.sqrt(sum)
            ecma = self.excima0 * np.exp(-(t / self.lamda))
            if d < ecma:
                neighbours.append(self.nodes[i])

        return neighbours

    def learn(self, input_data):
        for epoch in range(self.epochs):
            for example in range(len(input_data)):
                print(f"epoch: {epoch}: {example}")
                for t in range(1, len(input_data)):
                    n = self.calc_cluster(input_data[example])
                    for j in range(self.N * self.M):
                        if j == n:
                            self.potential[j] -= self.p_min
                        else:
                            self.potential[j] += (1 / self.N)
                    gaol_node = self.nodes[n]

                    sigma = self.excima0 * np.exp(-(t / self.lamda))
                    l = self.L0 * np.exp(-(t / self.lamda))

                    neighbours = self.get_neighbours(gaol_node, t)

                    for i in range(len(neighbours)):

                        d = np.sqrt((neighbours[i].x - gaol_node.x) ** 2 +
                                    (neighbours[i].y - gaol_node.y) ** 2)

                        theta = np.exp(-((d * d) / (2 * (sigma * sigma))))

                        for j in range(0, len(neighbours[i].weights)):
                            weight = neighbours[i].weights[j]
                            neighbours[i].weights[j] += theta * l * (input_data[example][j] - weight)

            self.nodes.sort(key=lambda x: self.get_sum_weights(x))
            self.update_color()

    def get_sum_weights(self, node):
        print(node)
        sum = 0
        for z in range(len(node.weights)):
            sum += node.weights[z]
        return sum

    def update_color(self):
        for i in range(self.N * self.M):
            node = self.nodes[i]
            sum1 = 0
            for z in range(len(node.weights)):
                sum1 += node.weights[z]
            node.color = sum1
