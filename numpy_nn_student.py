#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Legeard Hugo and Esterbet Julien
"""
#%%

import struct
import numpy as np
import matplotlib.pyplot as plt
import timeit



# provided function for reading idx files
def read_idx(filename):
    '''Reads an idx file and returns an ndarray'''
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Task 1: reading the MNIST files into Python ndarrays
x_test = read_idx("/Users/hugolegeard/Downloads/Fac/M1_EIT/S7/BDA/TP/TP2/mnist/t10k-images.idx3-ubyte")

y_test = read_idx("/Users/hugolegeard/Downloads/Fac/M1_EIT/S7/BDA/TP/TP2/mnist/t10k-labels.idx1-ubyte")

x_train = read_idx("/Users/hugolegeard/Downloads/Fac/M1_EIT/S7/BDA/TP/TP2/mnist/train-images.idx3-ubyte")

y_train = read_idx("/Users/hugolegeard/Downloads/Fac/M1_EIT/S7/BDA/TP/TP2/mnist/train-labels.idx1-ubyte")

       
# Task 2: visualize a few bitmap images
plt.imshow(x_train[2])



# Task 3: input pre-preprocessing    

x_trainProcessed= x_train.reshape(60000, 784)/255
x_testProcessed = x_test.reshape(10000, 784)/255


# Task 4: output pre-processing
def outputPreProcessing(filename) :
    res = np.zeros((len(filename), 10), dtype=int)
    res[np.arange(len(filename)), filename] = 1
    return res

y_trainProcessed = outputPreProcessing(y_train)

# Task 5-6: creating and initializing matrices of weights
def layer_weight(m,n) :
    return np.random.normal(0, 1/np.sqrt(n), (m,n))

w1 = layer_weight(784, 128)
w2 = layer_weight(128, 64)
w3 = layer_weight(64, 10)



# Task 7: defining functions sigmoid, softmax, and sigmoid'
def sigmoid(x) :
    return 1/(1+np.exp(-x))

def softmax(x) :
    return np.exp(x-np.max(x))/ np.sum(np.exp(x-np.max(x)))

def derivative_sigmoid(x) :
    return np.exp(-x)/((np.exp(-x)+1)**2)
    
# Task 8-9: forward pass
def forward_pass(x) :
    z1 = np.dot(x, w1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2)
    a2 = sigmoid(z2)
    z3 = np.dot(a2, w3)
    a3 = softmax(z3)
    return z1, a1, z2, a2, a3

# Task 11: weight updates
def weight_update(w1, w2, w3, deltaW1, deltaW2, deltaW3) :
	w1 -= 0.001*deltaW1
	w2 -= 0.001*deltaW2
	w3 -= 0.001*deltaW3
	return w1, w2, w3


# Task 10: backpropagation
def backpropagation(y,x) :
    z1, a1, z2, a2, a3 = forward_pass(x)
    e3 = a3 - y
    deltaW3 = np.outer(a2.T,e3)
    e2 = np.multiply(((e3).dot(w3.T)), derivative_sigmoid(z2))
    deltaW2 = np.outer(a1.T,e2)
    e1 = np.multiply(((e2).dot(w2.T)), derivative_sigmoid(z1))
    deltaW1 = np.outer(x.T, e1)
    return weight_update(w1, w2, w3, deltaW1, deltaW2, deltaW3)


# Task 12: computing error on test data
def compute_error(x, y) :
	z1, a1, z2, a2, a3 = forward_pass(x)
	return np.mean(np.argmax(a3, axis=1) != np.argmax(y, axis=1)) * 100
    
# Task 13: error with initial weights
#print(compute_error(x_trainProcessed, y_trainProcessed))
"""Yes Because the weights are random so we have 1 chance out of 10 to have the right answer"""
        
# Task 14-15: training
def train(x, y, nbEpoch) :
    for i in range(nbEpoch) :
        start = timeit.default_timer()
        for j in range(len(x)) :
            backpropagation(y[j], x[j])
        stop = timeit.default_timer()
        print("Epoch : ", i, "Error : ", compute_error(x, y), "Time : ", stop-start, "seconds")

#train(x_trainProcessed, y_trainProcessed, 30)


# Task 16-18: batch training


#%%