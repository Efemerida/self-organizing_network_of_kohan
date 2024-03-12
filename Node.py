class Node:
    
    def __init__(self, x, y, color_R, color_G, color_B):
        self.x = x
        self.y = y
        self.color_R = color_R
        self.color_G = color_G
        self.color_B = color_B

    def __str__(self):
        return (f"({self.x};{self.y}) {self.color_R}:{self.color_G}:{self.color_B}")

