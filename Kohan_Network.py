import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import imageio

import utils


def normalization(data):
    normalisation = []

    for i in range(len(data)):
        normalisation.append([])
        sumPow = 0
        for j in range(len(data[0])):
            sumPow += data[i][j] ** 2
        sqrt = np.sqrt(sumPow)
        for j in range(len(data[0])):
            normalisation[i].append(data[i][j] / sqrt)
    return normalisation


class Node:
    def __init__(self, x, y, weights):
        self.x = x
        self.y = y
        self.weights = weights
        rgb = np.random.random(3)
        self.color = [1, 1, 1]


class KohanaNetwork:
    M = N = 0
    nodes = []
    input_data = []
    alfa = 0.1
    epochs = 3
    p_min = 0.8
    T = 3
    L0 = 0.3
    lamda = 0

    frames = []

    excima0 = 0
    potential = []

    def normalization(self, data):

        normalisation = []

        for i in range(len(data)):
            normalisation.append([])
            sumPow = 0
            for j in range(len(data[0])):
                sumPow += data[i][j] ** 2
            sqrt = np.sqrt(sumPow)
            for j in range(len(data[0])):
                normalisation[i].append(data[i][j] / sqrt)
        return normalisation

    def __init__(self, N, M, input_data=None):
        self.M = M
        self.N = N
        self.input_data = input_data
        self.potential = [[1 / N for _ in range(N)] for _ in range(M)]
        self.excima0 = max(N, M) / 2
        self.lamda = self.T / np.log(self.excima0)

    def init_weights(self):
        for i in range(self.N):
            self.nodes.append([])
            for j in range(self.M):
                f_n = 0.03
                t_n = 0.8
                weights = [0, 0, 0, 0]
                tt = np.random.randint(3, 80, (1, 4))
                for p in range(4):
                    weights[p] = tt[0][p] / 100
                self.nodes[i].append(Node(i, j, weights))
                print(self.nodes[i][j].weights)
    def calc_d(self, node1, node2):
        return np.sqrt((node2.x - node1.x) ** 2 +
                       (node2.y - node1.y) ** 2)


    def ecma(self, t):
        return self.excima0 * np.exp(-(t / self.lamda))

    def calc_L(self, t):
        return self.L0 * np.exp(-(t / self.lamda))

    def calc_O(self, r, t):
        ecma_t = self.ecma(t)
        return np.exp(-((r * r) / (2 * ecma_t * ecma_t)))

    def calc_claster(self, input_data):
        d = np.zeros((self.N, self.M))
        for i in range(self.N):
            for j in range(self.M):
                for z in range(len(input_data) - 1):
                    d[i][j] += ((input_data[z] - self.nodes[i][j].weights[z]) ** 2)

        return np.argmin(d)

    def get_neighbours(self, node, t):
        neighbours = []
        for i in range(self.N):
            for j in range(self.M):
                d = self.calc_d(node, self.nodes[i][j])
                if d < self.ecma(t):
                    neighbours.append(self.nodes[i][j])

        return neighbours

    def update_color(self):
        for i in range(self.N):
            for j in range(self.M):
                node = self.nodes[i][j]
                color = [1, 1, 1]
                sum1 = 0
                sum2 = 0
                sum3 = 0
                # for k in range(len(self.input_data[0])):
                #     if k < 11:
                #         sum1 += node.weights[k] * 10000000000
                #     elif 11 <= k < 22:
                #         sum2 += node.weights[k] * 10000000000
                #     else:
                #         sum3 += node.weights[k] * 10000000000
                # print(node.weights)
                color[0] *= node.weights[0]
                color[1] *= node.weights[1]
                color[2] *= (node.weights[2] + node.weights[3]) % 1
                node.color = color

    def learn(self, input_data):
        for epoch in range(self.epochs):
            for example in range(len(input_data)):
                print(f"epoch: {epoch}: {example}")
                for t in range(1, len(input_data)):

                    n = self.calc_claster(input_data[example])
                        # for j in range(self.N):
                        #     for k in range(self.M):
                        #         if j == n:
                        #             self.potential[j][k] -= self.p_min
                        #         else:
                        #             self.potential[j][k] += (1 / self.N)
                    gaol_node = self.nodes[n // self.N][n % self.N]

                    sigma = self.ecma(t)
                    L = self.calc_L(t)

                    neighbours = self.get_neighbours(gaol_node, t)

                    for i in range(0, len(neighbours)):
                        r = self.calc_d(neighbours[i], gaol_node)
                        theta = np.exp(-((r**2)/(2*(sigma**2))))
                        for j in range(0, len(neighbours[i].weights)):
                            weight = neighbours[i].weights[j]
                            neighbours[i].weights[j] += theta * L * (input_data[example][j] - weight)
                    #print(gaol_node.weights)

                    # for i in range(self.N):
                    #     for j in range(self.M):
                    #         d = self.calc_d(gaol_node, self.nodes[i][j])
                    #         if d < self.ecma(t):
                    #             for z in range(len(input_data[example])):
                    #                 self.nodes[i][j].weights[z] += (self.calc_O(d, t) * self.calc_L(t)
                    #                                                 * (input_data[example][z] - self.nodes[i][j].weights[z]))

                # self.update_color()
                # figure, axes = plt.subplots()
                # L0 = 0.33
                # for i in self.nodes:
                #     for node in i:
                #         self.update_color()
                #         color = {'r': node.color[0], 'g': node.color[1], 'b': node.color[2]}
                #         circle = plt.Circle((node.x, node.y,), 0.3, fill=True,
                #                             color=(color['r'], color['g'], color['b']))
                #         axes.set_aspect(1)
                #         axes.add_artist(circle)
                #
                # plt.title('Circle')
                # plt.xlim(-1, 10)
                # plt.ylim(-1, 10)
                #
                # plt.show()

            self.update_color()

        # if z <= 261:
        #     r += self.nodes[i][j].weights[z] * z
        # elif 261 <= z <= 522:
        #     g += self.nodes[i][j].weights[z] * z
        # else:
        #     b += self.nodes[i][j].weights[z] * z

        # print(f'count {count}')
        #         self.create_frame(t)
        #
        # for i in range(1, self.T):
        #     self.frames.append(imageio.imread(f'img_{t}.png'))
        #
        # imageio.mimsave('./example.gif',  # output gif
        #                 self.frames,  # array of input frames
        #                 fps=1)

    # def create_frame(self, t):
    #     figure, axes = plt.subplots()
    #
    #     for i in self.nodes:
    #         for node in i:
    #             color = {'r': node.color[0], 'g': node.color[1], 'b': node.color[2]}
    #             circle = plt.Circle((node.x, node.y,), 0.3, fill=True,
    #                                 color=(color['r'], color['g'], color['b']))
    #             axes.set_aspect(1)
    #             axes.add_artist(circle)
    #
    #     plt.title('Circle')
    #     plt.xlim(-1, 10)
    #     plt.ylim(-1, 10)
    #     plt.savefig(f'img_{t}.png',
    #                 transparent=False,
    #                 facecolor='white'
    #                 )
    #     plt.close()


images, label = utils.load_dataset()

iris_dataset = datasets.load_iris()

input_d = images[0]
dataSet = iris_dataset["data"]
# data = np.genfromtxt('fruit.csv', delimiter=';')
# dataSet = []
# for i in range(1, len(data)):
#     tmp = []
#     dataSet.append(tmp)
#     for j in range(33):
#         dataSet[i - 1].append(data[i][j])

# wineSet = datasets.load_wine()
# set = [(wineSet.data[i][None, ...], wineSet.target[i]) for i in range(len(wineSet))]
#
# dataset = [set[i][0][0] for i in range(len(set))]
# print(set)
nDataset = normalization(dataSet)
print(nDataset)
# color  = [1, 1, 1]
# for k in range(len(nDataset[0])):
#     sum += nDataset[0][k] * 10000000000
# color[0] = np.abs(sum % 256) / 255 % 1
# color[1] = np.abs(sum % 256) / 255 % 1
# color[2] = np.abs(sum % 256) / 255 % 1
# figure, axes = plt.subplots()
#
# color = {'r': color[0], 'g': color[1], 'b': color[2]}
# circle = plt.Circle((5.0, 5.0,), 0.3, fill=True,
#                             color=(color['r'], color['g'], color['b']))
# axes.set_aspect(1)
# axes.add_artist(circle)
kn = KohanaNetwork(10, 10, nDataset)

kn.init_weights()
# print(kn.calc_claster(images[0]))
kn.learn(nDataset)

figure, axes = plt.subplots()
L0 = 0.33
for i in kn.nodes:
    for node in i:
        kn.update_color()
        color = {'r': node.color[0], 'g': node.color[1], 'b': node.color[2]}
        print(color)

        circle = plt.Circle((node.x, node.y,), 0.3, fill=True,
                            color=(color['r'], color['g'], color['b']))
        axes.set_aspect(1)
        axes.add_artist(circle)

# for i in kn.nodes:
#     for node in i:
#         print(f"node {i}:{node}")
#         print(node.weights)
plt.title('Circle')
plt.xlim(-1, 10)
plt.ylim(-1, 10)

plt.show()
