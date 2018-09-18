import numpy as np

# input data dimensions: x(d,1)
d = 784

# output labels: y(k,1)
k = 10

# 单层全连接神经网络
# Input layer: (d, 1), Output layer: (k, 1)
# Weight Matrix: (k, d), bias vector: (k, 1) => \theta: (k, d+1)
# Model: h = W * x + b

X = np.ndfromtxt('images.csv', delimiter=',')
y_orig = np.ndfromtxt("labels.csv", delimiter=',', dtype=np.int8)
img_size = X.shape[1]
num_class = 10
Y = np.zeros([len(y_orig), num_class])
Y[np.arange(len(y_orig)), y_orig] = 1

#np.insert(X, 0, values=np.ones(3), axis=1)


def softmax(X):
    x = np.exp(X)
    p = np.sum(x, axis=1, keepdims=True)
    return x / p


# 初始化W: weights, b: bias using normal distribution
W = np.random.normal(loc=0, scale=0.01, size=(k, d))
b = np.zeros(shape=(k, 1))

theta = np.random.normal(loc=0, scale=0.01, size=(k, d+1))

# X维度从d扩展到d+1，其中，第0维的值为1
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)


# theta: (k, d+1)
# xi: d+1 vector, x0 = 1
def h_elmentwise(theta, xi):
    e = np.exp(np.dot(theta, xi))
    return e / np.sum(e)


# theta: (k, d+1)
# X: (m, d+1), m is the number of samples. d is the dimension of input x. the extra one dimension is always be 1
# return: (k, m)
def h_vec(theta, X):
    x = np.exp(np.matmul(theta, X.T))
    return x / np.sum(x, axis=0, keepdims=True)


def sgd_vec(X_train, Y_train, theta, lamda, alpha):
    # mini batch sgd
    # 小批量随机梯度下降
    y_hat = h_vec(theta, X_train)
    diff = y_hat - Y_train
    for i in range(theta.shape[0]):
        theta[i, :] = theta[i, :] - alpha * (np.matmul(np.reshape(diff[:, i], [1, -1]), X_train) / X_train.shape[0] + lamda * theta[i, :])
    return theta


def train(X_train, Y_train, theta, num_epochs, alpha, lamda):
    for i in range(num_epochs):
        sgd_vec(X_train, Y_train=Y_train, theta=theta, alpha=alpha, lamda=lamda)
    return theta


def accuracy():
    return None


def predict(X, theta):
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
    predict_y = h_vec(theta=theta, X=X)
    print(predict_y)
    return predict_y


def predic_y(X, theta):
    """
    预测标签y

    :param X: 输入数据, shape=(m,d), m为样本数, d为维度
    :param theta: 参数矩阵
    :return:
    """
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
    return np.argmax(h_vec(theta, X), axis=1)


# 定义网络层
def net(X, W, b):
    return softmax(np.matmul(X, W) + b)


# 交叉熵损失函数
def cross_entropy(Y, Y_hat):
    M = Y.shape[0]
    y = np.choose(Y, Y_hat.T)
    return -(1/M) * np.sum(np.log(y))


# shape: (k,m)
def ground_true():
    return None


# shape: (k,m)
def proboblility_matrix():
    return None


def predict_p(W, b, X):
    a = np.exp(np.matmul(W, X) + b)
    return a / np.sum(a, axis=0, keepdims=True)


