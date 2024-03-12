import numpy as np
import matplotlib.pyplot as plt

import utils


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
    alfa = 0.8
    epochs = 1
    p_min = 0.1
    T = 100
    L0 = 0.3

    excima0 = 0
    potential = []

    def __init__(self, N, M, input_data=None):
        self.M = M
        self.N = N
        self.input_data = input_data
        self.potential = [[1 / N for _ in range(N)] for _ in range(M)]
        self.excima0 = max(N, M) / 2


    def init_weights(self):
        for i in range(self.N):
            self.nodes.append([])
            for j in range(self.M):
                self.nodes[i].append(Node(i, j, np.random.random(784)))

    def winner(self):
        d = [0] * self.N
        for i in d:
            d[i] = [0] * self.M
        for i in range(self.N):
            for j in range(self.M):
                for z in range(len(self.input_data[0])):
                    d[i][j] += (self.input_data[z] - self.nodes[i][j][z]) ** 2
        return np.argmin(d)

    def calc_d(self, node1, node2):
        return np.sqrt((node1.x - node2.x) ** 2 +
                       (node1.y - node2.y) ** 2)

    def lamda(self):
        return self.T/np.log(self.excima0)

    def excima(self, t):
        return self.excima0 * np.exp(-t/self.lamda())

    def calc_L(self, t):
        return self.L0 * np.exp(-t/self.lamda())

    def calc_O(self, r, t):
        return np.exp(-(r**2)/(2*self.excima(t)**2))

    def update(self):
        n = self.winner()
        for i in range(self.M):
            self.weights[n][i] += self.alfa * (self.input_data[i] - self.weights[n][i])

    def calc_claster(self, input_data):
        d = np.zeros((self.N, self.M))
        for i in range(self.N):
            for j in range(self.M):
                for z in range(len(input_data)-1):
                    d[i][j] += ((input_data[z] - self.nodes[i][j].weights[z]) ** 2)

        return np.argmin(d)

    def learn(self, input_data):
        for epoch in range(self.epochs):
            for t in range(1, self.T):
                print(f"epoch: {epoch}: {t % 100}")
                n = self.calc_claster(input_data[t])

                for j in range(self.N):
                    for k in range(self.M):
                        if j == n:
                            self.potential[j][k] -= self.p_min
                        else:
                            self.potential[j][k] += (1 / self.N)
                gaol_node = self.nodes[n//10][n%10]

                for i in range(self.N):
                    for j in range(self.M):
                        d = self.calc_d(gaol_node, self.nodes[i][j])
                        if d < self.excima(t):
                            r = 0
                            g = 0
                            b = 0
                            for z in range(len(input_data[t])):
                                self.nodes[i][j].weights[z] += self.calc_O(d, t) * self.calc_L(t) * (input_data[t][z] - self.nodes[i][j].weights[z])
                                if z <= 261:
                                    r += self.nodes[i][j].weights[z] * z
                                elif 261 <= z <= 522:
                                    g += self.nodes[i][j].weights[z] * z
                                else:
                                    b += self.nodes[i][j].weights[z] * z

                            self.nodes[i][j].color[0] = (r % 255)/255
                            self.nodes[i][j].color[1] = (g % 255)/255
                            self.nodes[i][j].color[2] = (b % 255)/255


images, label = utils.load_dataset()

input_d = images[0]

kn = KohanaNetwork(10, 10, input_d)
kn.init_weights()
print(kn.calc_claster(images[0]))
kn.learn(images)

figure, axes = plt.subplots()
L0 = 0.8

for i in kn.nodes:
    for node in i:
       color = {'r': node.color[0], 'g':  node.color[1], 'b': node.color[2]}
       circle = plt.Circle((node.x, node.y,), 0.3, fill=True,
                           color=(color['r'], color['g'], color['b']))
       axes.set_aspect(1)
       axes.add_artist(circle)


for i in kn.nodes:
    for node in i:
        print(f"node {i}:{node}")
        print(node.weights)
plt.title('Circle')
plt.xlim(-1, 10)
plt.ylim(-1, 10)

plt.show()
