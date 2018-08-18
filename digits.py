from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

# Load the MNIST digit data
M = loadmat("mnist_all.mat")
figures_folder = "figures/"

def get_test(M):
    batch_xs = np.zeros((0, 28 * 28))
    batch_y_s = np.zeros((0, 10))

    test_k = ["test" + str(i) for i in range(10)]
    for k in range(10):
        batch_xs = np.vstack((batch_xs, ((np.array(M[test_k[k]])[:]) / 255.)))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (len(M[test_k[k]]), 1))))
    return batch_xs.T, batch_y_s.T


def get_train(M):
    batch_xs = np.zeros((0, 28 * 28))
    batch_y_s = np.zeros((0, 10))
    validation_xs = np.zeros((0, 28 * 28))
    validation_y_s = np.zeros((0, 10))

    train_k = ["train" + str(i) for i in range(10)]
    for k in range(10):
        k_size = len(M[train_k[k]])
        train_size = (80 * k_size) / 100
        validation_size = k_size - train_size

        data = np.array(M[train_k[k]]) / 255.
        batch_xs = np.vstack((batch_xs, (data[:train_size])))
        validation_xs = np.vstack((validation_xs, (data[train_size:])))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (train_size, 1))))
        validation_y_s = np.vstack((validation_y_s, np.tile(one_hot, (validation_size, 1))))
    return batch_xs.T, batch_y_s.T, validation_xs.T, validation_y_s.T


train_x, train_y, validation_x, validation_y = get_train(M)
test_x, test_y = get_test(M)
np.random.seed(0)
W0 = np.random.uniform(-0.01, 0.01, (784, 10))
b0 = np.random.uniform(-0.01, 0.01, (10, 1))
alpha = 1e-5
W_part6 = np.ndarray(1)

def softmax(y):
    return exp(y) / tile(sum(exp(y), 0), (len(y), 1))

def forward(x, W0, b0):
    L0 = dot(W0.T, x) + b0
    output = softmax(L0)
    return L0, output

def get_performance(x, y, W0, b0):
    L0, output = forward(x, W0, b0)
    expected = np.array([i.tolist().index(1) for i in y.T])
    actual = np.array([i.tolist().index(max(i)) for i in output.T])
    results = (expected == actual)
    true_num = results.tolist().count(True)
    performance = true_num / float(output.shape[1])
    return performance * 100

def cost(x, W0, b0, y):
    L0, p = forward(x, W0, b0)
    while p.min() < 1e-323:
        ind = np.unravel_index(np.argmin(p, axis=None), p.shape)
        p[ind] = 1e-323
    return -np.sum(y * np.log(p))

def get_prediction(W0, b0, x):
    x = test_x
    L0, output = forward(x, W0, b0)
    y = np.argmax(output, 0)
    print y
    return y

# gradient of the cost function with respect to the weights
def dw_cost(x, W0, b0, y):
    L0, p = forward(x, W0, b0)
    dCdL0 = p - y
    dCdW0 = dot(x, dCdL0.T)
    return dCdW0

# gradient of the cost function with respect to the biases
def db_cost(x, W0, b0, y):
    L0, p = forward(x, W0, b0)
    dCdL0 = p - y
    dCdb0 = np.sum(dCdL0, 1)
    return dCdb0.reshape(10, 1)

def test_gradient(W0, b0, x, y):
    h = 1e-5

    c1 = cost(x, W0, b0, y)
    W02 = W0.copy()
    W02[127][0] += h
    c2 = cost(x, W02, b0, y)
    diff = (c2 - c1) / h
    dc = dw_cost(x, W0, b0, y)
    print "weight gradient1:", dc[127][0], diff
    W02 = W0.copy()
    W02[272][7] += h
    c2 = cost(x, W02, b0, y)
    diff = (c2 - c1) / h
    dc = dw_cost(x, W0, b0, y)
    print "weight gradient2:", dc[272][7], diff
    
    b02 = b0.copy()
    b02[0] += h
    c2 = cost(x, W0, b02, y)
    diff = (c2 - c1) / h
    dc = db_cost(x, W0, b0, y)
    print "bias gradient1:", dc[0], diff
    b02 = b0.copy()
    b02[8] += h
    c2 = cost(x, W0, b02, y)
    diff = (c2 - c1) / h
    dc = db_cost(x, W0, b0, y)
    print "bias gradient2:", dc[8], diff

def gradient_descent(x, y, validation_x,validation_y, init_W, b0, alpha, max_iter = 2500, momentum = 0):
    EPS = 1e-5  # EPS = 10**(-5)
    prev_W = init_W - 10 * EPS
    W0 = init_W.copy()

    if momentum != 0:
        v = 1
        vb = 1
        name = "part5"
    else:
        name = "part4"

    results = {}
    iter = 0
    while norm(W0 - prev_W) > EPS and iter < max_iter:
        prev_W = W0.copy()
        c0 = cost(x, W0, b0, y)
        results[iter] = [get_performance(x, y, W0, b0), get_performance(validation_x, validation_y, W0, b0)]
        new_W = dw_cost(x, W0, b0, y)
        new_b = db_cost(x, W0, b0, y)
        if momentum != 0:
            v = momentum * v + (alpha * new_W)
            vb = momentum * vb + (alpha * new_b)
            W0 -= v
            b0 -= vb
        else:
            W0 -= (alpha * new_W)
            b0 -= (alpha * new_b)
        if iter % 100 == 0:
            print "Iter", iter
            print "cost = ", c0
        iter += 1
    print "Iter", iter
    print "cost = ", c0
    learning_curve(results, name)
    if momentum == 0:
        W_images(W0, name)
    return W0

def W_images(W0, name):
    plt.close('all')
    np.random.seed(0)
    f, axarr = plt.subplots(1, 10)
    for i in range(10):
        im = np.reshape(W0[:,i], (28, 28))
        axarr[i].imshow(im)
        axarr[i].set_title('W' + str(i))
    plt.setp([a.get_xticklabels() for a in axarr[:]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:]], visible=False)
    f.tight_layout()
    f.savefig(figures_folder + name + "_weights.png")
    plt.close('all')

def learning_curve(results, name):
    plt.close('all')
    lists = sorted(results.items())
    x, y = zip(*lists)

    plt.figure(int(name[-1]))
    plt.plot(x, y)
    plt.xlabel("Epochs")
    plt.ylabel("Performance %")
    plt.legend(["Training", "Validation"])
    plt.savefig(figures_folder + name + "_learning_curve.png")
    plt.close('all')

def gradient_descent_w1_w2(w_1, w_2, x, y, validation_x,validation_y, init_W, b0, alpha, max_iter = 2500, momentum = 0):
    EPS = 1e-5  # EPS = 10**(-5)
    prev_W = init_W - 10 * EPS
    W0 = init_W.copy()
    r = []

    if momentum != 0:
        v = 1
        name = "part5"
    else:
        name = "part4"

    results = {}
    iter = 0
    while iter < max_iter:
        prev_W = W0.copy()
        c0 = cost(x, W0, b0, y)
        results[iter] = [get_performance(x, y, W0, b0), get_performance(validation_x, validation_y, W0, b0)]
        new_W = dw_cost(x, W0, b0, y)
        r.append((W0[w_1[0], w_1[1]], W0[w_2[0], w_2[1]]))
        if momentum != 0:
            v = momentum * v + (alpha * new_W)
            W0[w_1[0], w_1[1]] -= v[w_1[0], w_1[1]]
            W0[w_2[0], w_2[1]] -= v[w_2[0], w_2[1]]
        else:
            W0[w_1[0], w_1[1]] -= (alpha * new_W)[w_1[0], w_1[1]]
            W0[w_2[0], w_2[1]] -= (alpha * new_W)[w_2[0], w_2[1]]

        if iter % 100 == 0:
            print "Iter", iter
            print "cost = ", c0
        iter += 1
    print "Iter", iter
    print "cost = ", c0
    return r

def get_contour(w_1, w_2, x, y, validation_x, validation_y, W2, b0):
    plt.close('all')
    W3 = W2.copy()
    new_W_1 = 25.
    new_W_2 = -25.

    W3[w_1[0], w_1[1]] = new_W_1
    W3[w_2[0], w_2[1]] = new_W_2
    alpha = 1e-4
    gd_traj = gradient_descent_w1_w2(w_1, w_2, x, y, validation_x, validation_y, W3, b0, alpha, 10)

    alpha = 1e-4
    W3 = W2.copy()
    W3[w_1[0], w_1[1]] = new_W_1
    W3[w_2[0], w_2[1]] = new_W_2
    mo_traj = gradient_descent_w1_w2(w_1, w_2, x, y, validation_x, validation_y, W3, b0, alpha, 10, 0.7)

    W3 = W2.copy()
    w1s = np.arange(-50, 50, 10)
    w2s = np.arange(-80, 80, 16)
    w1z, w2z = np.meshgrid(w1s, w2s)
    C = np.zeros([w1s.size, w2s.size])
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            W3 = W2.copy()
            W3[w_1[0], w_1[1]] = w1
            W3[w_2[0], w_2[1]] = w2
            C[i, j] = cost(x, W3, b0, y)
    plt.figure(6)
    CS = plt.contour(w1z, w2z, C)
    plt.plot([a for a, b in gd_traj], [b for a, b in gd_traj], 'yo-', label="No Momentum")
    plt.plot([a for a, b in mo_traj], [b for a, b in mo_traj], 'go-', label="Momentum")
    plt.legend(loc='top left')
    plt.title('Contour plot')
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.savefig(figures_folder + "part6_Contour_plot.png")
    plt.close('all')

def part1():
    plt.close('all')
    np.random.seed(0)
    f, axarr = plt.subplots(10, 10)
    for i in range(10):
        size = M["train" + str(i)].shape[0]
        for j in range(10):
            indx = np.random.randint(0, (size - 1))
            im = M["train" + str(i)][indx, :].reshape((28,28))
            axarr[i, j].imshow(im)
        plt.setp([a.get_xticklabels() for a in axarr[:, i]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, i]], visible=False)
    f.savefig(figures_folder + "part1_digits.png")
    plt.close('all')

def part2():
    get_prediction(W0, b0, test_x)

def part3():
    x = train_x[:, 0]
    x = x.reshape(784, 1)
    y = train_y[:,0]
    y = y.reshape(10, 1)
    test_gradient(W0, b0, x, y)

def part4():
    init_W = W0.copy()
    init_b = b0.copy()
    W1 = gradient_descent(train_x, train_y, validation_x,validation_y, init_W, init_b, alpha, 1000)
    print "train performance:", get_performance(train_x, train_y, W1, init_b)
    print "validation performance:", get_performance(validation_x, validation_y, W1, init_b)
    print "test performance:", get_performance(test_x, test_y, W1, init_b)

def part5():
    init_W = W0.copy()
    init_b = b0.copy()
    W2 = gradient_descent(train_x, train_y, validation_x,validation_y, init_W, init_b, alpha, 300, 0.7)
    print "train performance:", get_performance(train_x, train_y, W2, init_b)
    print "validation performance:", get_performance(validation_x, validation_y, W2, init_b)
    print "validation performance:", get_performance(test_x, test_y, W2, init_b)
    global W_part6
    W_part6 = W2.copy()

def part6():
    if W_part6.shape == (1,):
        print "Please run part5 first!"
        return
    init_W = W_part6.copy()
    init_b = b0.copy()
    dw = dw_cost(train_x, init_W, b0, train_y)
    w_1 = [np.unravel_index(dw.argmax(), init_W.shape)[0], np.unravel_index(dw.argmax(), init_W.shape)[1]]
    w_2 = [np.unravel_index(dw.argmin(), init_W.shape)[0], np.unravel_index(dw.argmin(), init_W.shape)[1]]

    get_contour(w_1, w_2, train_x, train_y, validation_x, validation_y, init_W, init_b)

# Main Function
if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()
    part6()
