import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

import utils


class Node:
    def __init__(self, x, y, weights):
        self.x = x
        self.y = y
        self.weights = weights
        self.color = 1


class Kohan_Network_2:
    nodes = []

    epochs = 10
    p_min = 0.4
    T = 15
    L0 = 0.2
    lamda = 0

    frames = []

    excima0 = 0
    potential = []

    def __init__(self, N, M, input_data):
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
                weights = [0] * len(self.input_data[0])
                tt = np.random.randint(3, 80, (1, 4))
                for p in range(4):
                    weights[p] = tt[0][p] / 100
                self.nodes[i].append(Node(i, j, weights))
                print(self.nodes[i][j].weights)

    def calc_cluster(self, input_data):
        d = np.zeros((self.N, self.M))
        for i in range(0, self.N):
            for j in range(0, self.M):
                if self.potential[i][j] < self.p_min:
                    d[i][j] = 2000
                    continue
                sum = 0
                for z in range(len(input_data)):
                    sum += ((input_data[z] - self.nodes[i][j].weights[z]) ** 2)
                d[i][j] = np.sqrt(sum)
        return np.argmin(d)

    def get_neighbours(self, node, t):
        neighbours = []
        ecma = self.excima0 * np.exp(-(t / self.lamda))
        for i in range(self.N):
            for j in range(self.M):
                d = np.sqrt((node.x - self.nodes[i][j].x) ** 2 +
                            (node.y - self.nodes[i][j].y) ** 2)
                if d < ecma:
                    neighbours.append(self.nodes[i][j])

        return neighbours

    def learn(self, input_data):
        for t in range(self.T):
            for data in range(1, len(input_data)):
                print(f"epoch: {t}: {data}")
                for example in range(len(input_data)):
                    n = self.calc_cluster(input_data[example])
                    for j in range(self.N):
                        for k in range(self.M):
                            if j == n:
                                self.potential[j][k] -= self.p_min
                            else:
                                self.potential[j][k] += (1 / self.N)
                    gaol_node = self.nodes[n // self.N][n % self.N]

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

            self.update_color()

    def update_color(self):
        for i in range(self.N):
            for j in range(self.M):
                node = self.nodes[i][j]
                sum = 0
                for z in range(len(node.weights)):
                    sum += node.weights[z]
                node.color = sum


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


class Kohan_Network:
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


iris_dataset = datasets.load_iris()["data"]
fruit_dataset = utils.get_fruit_dataset()

wine_dataset_file = np.genfromtxt('wine.csv', delimiter=',')
wine_dataset = utils.get_wine_dataset()

print(len(iris_dataset))
arr = []
p = 0
for i in range(50):
    arr.append([])
    for j in range(3):
        arr[i].append(sum(iris_dataset[p]))
        p += 1
plt.imshow(arr, cmap='viridis')
plt.show()

normalization_dataset = normalization(iris_dataset)
normalization_dataset = np.random.permutation(normalization_dataset)

kn = Kohan_Network(50, 3, normalization_dataset)
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
plt.close()
plt.imshow(arr, cmap='viridis')
plt.show()

# kn.learn(normalization_dataset)


# arr = []
#
# sum = 0
# z = 0
# p = 0
# for i in range(50):
#     arr.append([])
#     for j in range(3):
#         arr[i].append(kn.nodes[p].color)
#         p += 1
#
# print(arr)
# plt.close()
# plt.imshow(arr, cmap='viridis')
# plt.show()


kn2 = Kohan_Network_2(10, 10, normalization_dataset)
kn2.init_weights()
kn2.learn(normalization_dataset)
kn2.update_color()

arr = []

sum = 0
z = 0
p = 0
for i in range(10):
    arr.append([])
    for j in range(10):
        arr[i].append(kn2.nodes[i][j].color)

print(arr)
plt.close()
plt.imshow(arr, cmap='viridis')
plt.show()

# normalization_dataset = normalization(iris_dataset)
#
# kn = Kohan_Network(10, 10, normalization_dataset)
# kn.init_weights()
# kn.update_color()
# kn.learn(normalization_dataset)
#
#
# # nDataset = np.random.permutation(nDataset)
# arr = [[0] * 10] * 10
#
# sum = 0
# z = 0
# p = 0
# for i in range(10):
#     for j in range(10):
#         arr[i][j] = kn.nodes[p].color
#         p += 1
#
# print(arr)
# plt.imshow(arr, cmap='viridis')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
#
# class Node:
#     def __init__(self, x, y, weights):
#         self.x = x
#         self.y = y
#         self.weights = weights
#         rgb = np.random.random(3)
#         self.color = [1, 1, 1]
# class Kohan_Network:
#     nodes = []
#
#     alfa = 0.1
#     epochs = 3
#     p_min = 0.2
#     T = 10
#     L0 = 0.33
#     lamda = 0
#
#     frames = []
#
#     excima0 = 0
#     potential = []
#
#     def __init__(self, N, M, input_data):
#         self.M = M
#         self.N = N
#         self.input_data = input_data
#         self.potential = [[1 / N for _ in range(N)] for _ in range(M)]
#         self.excima0 = max(N, M) / 2
#         self.lamda = self.T / np.log(self.excima0)
#
#     def init_weights(self):
#         for i in range(self.N):
#             self.nodes.append([])
#             for j in range(self.M):
#                 weights = [0] * len(self.input_data[0])
#                 tt = np.random.randint(3, 80, (1, 4))
#                 for p in range(4):
#                     weights[p] = tt[0][p] / 100
#                 self.nodes[i].append(Node(i, j, weights))
#                 print(self.nodes[i][j].weights)
#
#     def calc_cluster(self, input_data):
#         d = np.zeros((self.N, self.M))
#         for i in range(self.N):
#             for j in range(self.M):
#                 if self.potential[i][j] < self.p_min: continue
#                 sum = 0
#                 for z in range(len(input_data)):
#                     sum += ((input_data[z] - self.nodes[i][j].weights[z]) ** 2)
#                 d[i][j] = np.sqrt(sum)
#
#         return np.argmin(d)
#
#     def get_neighbours(self, node, t):
#         neighbours = []
#         for i in range(self.N):
#             for j in range(self.M):
#                 d = np.sqrt((node.x - self.nodes[i][j].x) ** 2 +
#                         (node.y - self.nodes[i][j].y) ** 2)
#                 ecma = self.excima0 * np.exp(-(t / self.lamda))
#                 if d < ecma:
#                     neighbours.append(self.nodes[i][j])
#
#         return neighbours
#
#
#     def learn(self, input_data):
#         for epoch in range(self.epochs):
#             for example in range(len(input_data)):
#                 print(f"epoch: {epoch}: {example}")
#                 for t in range(1, len(input_data)):
#                     n = self.calc_cluster(input_data[example])
#                     for j in range(self.N):
#                         for k in range(self.M):
#                             if j == n:
#                                 self.potential[j][k] -= self.p_min
#                             else:
#                                 self.potential[j][k] += (1 / self.N)
#                     gaol_node = self.nodes[n // self.N][n % self.N]
#
#                     sigma = self.excima0 * np.exp(-(t / self.lamda))
#                     l = self.L0 * np.exp(-(t / self.lamda))
#
#                     neighbours = self.get_neighbours(gaol_node, t)
#
#                     for i in range(len(neighbours)):
#
#                         d = np.sqrt((neighbours[i].x - gaol_node.x) ** 2 +
#                                 (neighbours[i].y - gaol_node.y) ** 2)
#
#                         theta = np.exp(-((d * d) / (2 * (sigma * sigma))))
#
#                         for j in range(0, len(neighbours[i].weights)):
#                             weight = neighbours[i].weights[j]
#                             neighbours[i].weights[j] += theta * l * (input_data[example][j] - weight)
#
#
#             self.update_color()
#
#
#
#     def update_color(self):
#         for i in range(self.N):
#             for j in range(self.M):
#                 node = self.nodes[i][j]
#                 color = [1, 1, 1]
#                 sum1 = 0
#                 sum2 = 0
#                 sum3 = 0
#                 for z in range(len(node.weights)):
#                     sum1 += node.weights[z]
#                 color[0] = sum1
#                 color[1] *= node.weights[1]
#                 color[2] *= ((node.weights[2] + node.weights[3]) / 2) % 1
#                 node.color = color
#
# def normalization(data):
#     normalisation = []
#     for i in range(len(data)):
#         normalisation.append([])
#         sum_pow = 0
#         for j in range(len(data[0])):
#             sum_pow += data[i][j] ** 2
#         sqrt = np.sqrt(sum_pow)
#         for j in range(len(data[0])):
#             normalisation[i].append(data[i][j] / sqrt)
#     return normalisation
#
# iris_dataset = datasets.load_iris()["data"]
#
# normalization_dataset = normalization(iris_dataset)
#
# kn = Kohan_Network(10, 10, normalization_dataset)
# kn.init_weights()
# kn.update_color()
# kn.learn(normalization_dataset)
# kn.learn(normalization_dataset)
#
#
#
#
#
# # nDataset = np.random.permutation(nDataset)
# arr = [[0] * 10] * 10
#
# sum = 0
# z = 0
# p = 0
# for i in range(10):
#     for j in range(10):
#         arr[i][j] = kn.nodes[i][j].color[0]
# print(arr)
# plt.imshow(arr, cmap='viridis')
# plt.show()
#
#
#
#
#
#
