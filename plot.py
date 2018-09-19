import matplotlib.pyplot as plt
import numpy as np


def xyplot(x_vals, y_vals, name):
    plt.figure(num=1, figsize=(5, 2.5))
    plt.plot(x_vals, y_vals)
    plt.xlabel('x')
    plt.ylabel(name + '(x)')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (1 - np.exp(-2*x)) / (1 + np.exp(-2*x))

x = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])

y = tanh(x)

xyplot(x, y, 'sigmoid')
plt.show()
