import matplotlib.pyplot as plt
import numpy as np
import SOM
import utils

from Node import Node


def init_nodes():
    res = []
    for i in range(28):
        for j in range(28):
            rgb = np.random.random(3)
            res.append(Node(i, j, rgb[0], rgb[1], rgb[2]))

    return res


def main():
    T, Y = utils.load_dataset()
    print(T)
    m, n = 3000, len(T[0])

    weights = np.random.uniform(0, 1, (3, 784))

    epochs = 3
    alpha = 0.5

    for i in range(epochs):
        print(f"epochs {i}")
        for j in range(m):
            # training sample
            sample = T[j]

            # Compute winner vector
            J = SOM.winner(weights, sample)

            # Update winning vector
            weights = SOM.update(weights, sample, J, alpha)

        # classify test sample
    s = T[2000]
    s_n = np.argmax(Y[2000])
    J = SOM.winner(weights, s)
    print(f"{J} {s_n}")
    print(weights)



    # list = init_nodes()
    # figure, axes = plt.subplots()
    # L0 = 0.8
    #
    # for node in list:
    #     color = {'r': node.color_R, 'g':  node.color_G, 'b': node.color_B}
    #     circle = plt.Circle((node.x, node.y,), 0.3, fill=True,
    #                         color=(color['r'], color['g'], color['b']))
    #     axes.set_aspect(1)
    #     axes.add_artist(circle)
    #
    # plt.title('Circle')
    # plt.xlim(-1, 28)
    # plt.ylim(-1, 28)
    #
    # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]
    # for i in range(10):
    #     rnd = np.random.random_integers(low=0, high=5)
    #     node_tmp = 0
    #     del_min = 20000
    #     for i in range(len(list)):
    #         node = list[i]
    #         delta = np.sqrt((node.color_R - colors[rnd][0]) * (node.color_R - colors[rnd][0]) +
    #                        (node.color_G - colors[rnd][1]) * (node.color_G - colors[rnd][1]) +
    #                        (node.color_B - colors[rnd][2]) * (node.color_B - colors[rnd][2]))
    #         if delta < del_min:
    #             del_min = delta
    #             node_tmp = list[i]
    #     d = np.sqrt(())
    #
    # plt.show()











if __name__=='__main__':
    main()