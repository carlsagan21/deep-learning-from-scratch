import numpy as np
from utils import step_function, sigmoid, relu, softmax, img_show
from mnist import load_mnist

#
# img = x_train[0]
# label = t_train[0]
# print(label)
#
# print(type(img))
# img = img.reshape(28, 28)
# print(img.shape)
#
# img_show(img)

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

import pickle
def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

import time

x, t = get_data()
network = init_network()

start_time = time.time()

batch_size = 500
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i + batch_size])
    # if p == t[i]:
    #     accuracy_cnt += 1
    # else:
    #     print(t[i])
    #     img = x[i]
    #
    #     img = img.reshape(28, 28)
    #     img = img * 255
    #     img_show(img)
    #     break

end_time = time.time()
tdelta = end_time - start_time

print('Accuracy: ' + str(float(accuracy_cnt) / len(x)))
print('Time: ' + str(tdelta))

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_train, t_train, x_test, t_test

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size