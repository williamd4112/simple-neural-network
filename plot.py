import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(func, x_, t_, x_min, y_min, x_max, y_max, h):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = func(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Set3, alpha=0.8)

    # Plot also the training points
    plt.scatter(x_[:, 0], x_[:, 1], c=t_, s=25, edgecolors='#666777', cmap=plt.cm.Set3)
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

