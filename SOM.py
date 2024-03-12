import math
import numpy as np


class SOM:
    L0 = 0.8
    M = 784
    N = 10
    T = 20
    sigma0 = 0
    lamda = 0

    def winner(weights, sample):
        d = [0] * len(weights)
        for i in range(len(sample)):
            for j in range(len(weights)):
                d[j] += math.pow((sample[i] - weights[j][i]), 2)
        return np.argmax(d)

    def update(weights, sample, J, alpha):
        for i in range(len(weights[0])):
            weights[J][i] = weights[J][i] + alpha * (sample[i] - weights[J][i])
        return weights

    def calculate_sigma0(self):
        self.sigma0 = max(self.M, self.N) / 2

    def calc_lambda(self):
        self.lamda = self.T / np.log(self.sigma0)

    def sigma_t(self, t):
        return self.sigma0 * np.exp(-t / self.lamda)

    def calc_d(node1, node2):
        return math.sqrt((node1.x - node2.x) * (node1.x - node2.x) +
                         (node1.y - node2.y) * (node1.y - node2.y))

    def calc_l(self, t):
        return self.L0 * np.exp(-t / self.lamda)

    def calc_o(self, t, d):
        return np.exp(-(d ** 2) / (2 * (self.sigma_t(t) ** 2)))

    def update_weights(self, weights, t):
        new_weights = weights.copy()
        for i in range(len(weights)):
            new_weights[i] = weights[i] + self.calc_o(t, )
