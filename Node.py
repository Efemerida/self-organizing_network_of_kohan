class Node:
    
    def __init__(self, x, y, weights):
        self.x = x
        self.y = y
        self.color = 1
        self.weights = weights

    def __str__(self):
        return (f"({self.x};{self.y}) {self.color}")

