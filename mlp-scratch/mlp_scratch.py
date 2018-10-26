import numpy as np
import matplotlib.pyplot as plt

batch_size = 256

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(loc=0, scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)

W2 = np.random.normal(loc=0, scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)

params = [W1, b1, W2, b2]


def relu(X):
    return np.max(X, 0)


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(np.dot(X, W1) + b1)


def corr2d(X, K):
    h, w = K.shape
    Y = np.zeros(shape=(X.shape[0] - h + 1, X.shape[1] - h + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = np.sum(X[i:i+h, j:j+w] * K)

    return Y


def corr2d_multi_in(X, K):
    """
    多通道输入互相关计算 (二维卷积核)
    :param X:
    :param K:
    :return:
    """
    t = None

    for x, k in zip(X, K):
        y = corr2d(x, k)
        if t is None:
            t = y
        else:
            t = t + y
    return t


class Conv2D:
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


#X = np.array([[0, 1, 2], [3, 4, 5], [6, 7,8 ]])
#K = np.array([[0, 1], [2, 3]])
#print(corr2d(X, K))

X = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = np.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(corr2d_multi_in(X, K))

