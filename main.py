import matplotlib.patches as ptc
import matplotlib.patches as Patch
import matplotlib.pyplot as pyplot



def main():

    circle = ptc.Circle((0.2,0.2), 0.1)
    circle2 = ptc.Circle((0.4,0.2), 0.1)
    circle3 = ptc.Circle((0.6,0.2), 0.1)


    fig, ax = pyplot.subplots()
    ax.add_patch(circle)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    ax.set_aspect('equal')
    pyplot.show()











if __name__=='__main__':
    main()