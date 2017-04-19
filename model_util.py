import numpy as np

def softmax(a):
    e = np.exp(a)
    return e / e.sum(axis=1)[:, np.newaxis]

def sigmoid(a):
    return np.ones_like(a) / (np.ones_like(a) + np.exp(-a))

def d_sigmoid(a):
    return sigmoid(a) * (np.ones_like(a) - sigmoid(a)) 

class DifferentiableFunction(object):
    def __call__(self, x):
        raise NotImplementedError()
    def d(self, x):
        raise NotImplementedError()

class Sigmoid(DifferentiableFunction):
    def __call__(self, x):
        return sigmoid(x)
    def d(self, x):
        return d_sigmoid(x)

class Softmax(DifferentiableFunction):
    def __call__(self, x):
        return softmax(x)
    def d(self, x):
       return softmax(x) * (np.ones_like(x) - softmax(x))
