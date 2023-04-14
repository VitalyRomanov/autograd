import numpy as np

from backward_ops import np_sigmoid, np_relu
from tensor import Tensor, _make_op_output_tensor

one = Tensor(np.array(1.))
zero = Tensor(np.array(0.))


def sigmoid(x: Tensor):  # has a derivative
    return _make_op_output_tensor(np_sigmoid, x)
    # return 1. / (1. + (-x).exp())


def relu(x: Tensor):  # has a derivative
    return _make_op_output_tensor(np_relu, x)
    # return x.splice(x <= zero, zero)


def tanh(x: Tensor):  # has a derivative
    return x.tanh()


def log_softmax(x):  # autograd
    return x - x.exp().sum().log()


def cross_entropy_loss(logits, y_true):  # autograd
    assert logits.shape == y_true.shape, \
        f"Shape mismatch: {logits.shape} != {y_true.shape}, provide one hot encoded labels"

    logits = log_softmax(logits)
    loss = logits * y_true
    return -loss.sum(dim=-1)
