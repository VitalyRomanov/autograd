import numpy as np

from activations_and_loss import sigmoid, relu, log_softmax, cross_entropy_loss
from backprop import backprop
from finite_diff import finite_difference
from tensor import Tensor, float64, reset_grads, make_parameter, no_grad, TensorSlice


def within_tolerance(x1, x2, tol):
    den = np.abs(x1)
    if den.shape != ():
        den[den == 0] = 1.
    diff = np.abs(x1 - x2) / den
    return np.all(diff < tol).item()


# noinspection DuplicatedCode
def test_backward_ops(grad_error_tolerance=1e-4):
    # use float64, otherwise finite difference is wrong
    a = make_parameter(np.array([[1., 2.], [3., 4.], [5., 6.]]), dtype=float64)
    b = make_parameter(np.array([[3.], [2.], [1.]]), dtype=float64)

    # add
    backprop(a + b)  # (a + b)' = 1 + 1
    assert a.grad is not None
    assert b.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data))
    assert np.array_equal(b.grad, np.ones_like(b.data) * 2.)
    reset_grads(a, b)

    # sub
    backprop(a - b)  # (a - b)' = 1 - 1
    assert a.grad is not None
    assert b.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data))
    assert np.array_equal(b.grad, -np.ones_like(b.data, dtype=float64) * 2.)
    reset_grads(a, b)

    # mult
    backprop(a * b)  # (ab)' = a'b + ab' = b + a
    assert a.grad is not None
    assert b.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data) * b.data)
    assert np.array_equal(b.grad, np.ones_like(b.data) * a.data.sum(axis=1, keepdims=True))
    reset_grads(a, b)

    # true div
    backprop(a / b)  # (a/b)' = a'/b + a/b' = 1/b - a/b^2
    assert a.grad is not None
    assert b.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data) * 1 / b.data)
    assert np.array_equal(b.grad, -np.ones_like(b.data) * (a.data / (b.data**2)).sum(axis=1, keepdims=True))
    reset_grads(a, b)

    # transpose
    backprop(a.T)
    assert a.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data))
    reset_grads(a)

    # matmul
    backprop(a.T @ b)  # (a/b)' = a'/b + a/b' = 1/b - a/b^2
    assert a.grad is not None
    assert b.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data) * b.data)
    assert np.array_equal(b.grad, np.ones_like(b.data) * a.data.sum(axis=1, keepdims=True))
    reset_grads(a, b)

    # power
    backprop(a ** b)  # (a^b)' = b * a^(b-1) + a^b * ln a
    assert a.grad is not None
    assert b.grad is not None
    assert np.array_equal(a.grad, b.data * a.data ** (b.data - 1))
    assert np.array_equal(b.grad, np.sum(a.data ** b.data * np.log(a.data), axis=1, keepdims=True))
    reset_grads(a, b)

    # negative
    backprop(-a)  # (-a)' = -1
    assert a.grad is not None
    assert np.array_equal(a.grad, -np.ones_like(a.data))
    reset_grads(a)

    # exp
    backprop(a.exp())  # (e^a)' = e^a
    assert a.grad is not None
    assert np.array_equal(a.grad, np.exp(a.data))
    reset_grads(a)

    # log
    backprop(a.log())  # (log a)' = 1 / a
    assert a.grad is not None
    assert np.array_equal(a.grad, 1 / a.data)
    reset_grads(a)

    # tanh
    backprop(a.tanh())  # (tanh(a)' = 1 - tanh(a)^2)
    assert a.grad is not None
    assert within_tolerance(a.grad, 1 - np.tanh(a.data) ** 2, grad_error_tolerance)
    reset_grads(a)

    # sum
    backprop(a.sum())
    assert a.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data))
    reset_grads(a)

    # sum
    backprop(a.sum(dim=0))
    assert a.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data))
    reset_grads(a)

    # sum
    backprop(a.sum(dim=1))
    assert a.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data))
    reset_grads(a)

    # mean
    backprop(a.mean())
    assert a.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data) / a.data.size)
    reset_grads(a)

    # mean
    backprop(a.mean(dim=0))
    assert a.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data) / a.data.shape[0])
    reset_grads(a)

    # mean
    backprop(a.mean(dim=1))
    assert a.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data) / a.data.shape[1])
    reset_grads(a)

    # abs
    backprop(a.abs())
    assert a.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data))
    reset_grads(a)

    # reshape
    backprop(a.reshape((-1,)))
    assert a.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data))
    reset_grads(a)

    # __getitem__
    backprop(a[:2])
    grad_ = np.ones_like(a.data)
    grad_[2, :] = 0
    assert a.grad is not None
    assert np.array_equal(a.grad, grad_)
    reset_grads(a)

    # __getitem__
    backprop(a[:2, :2])
    grad_ = np.ones_like(a.data)
    grad_[2, :] = 0
    assert a.grad is not None
    assert np.array_equal(a.grad, grad_)
    reset_grads(a)

    # __getitem__
    backprop(a[Tensor(np.array([True, False, True]))])
    grad_ = np.ones_like(a.data)
    grad_[1, :] = 0
    assert a.grad is not None
    assert np.array_equal(a.grad, grad_)
    reset_grads(a)

    # splice
    target = make_parameter(np.array(-4.))
    ind = Tensor(np.array(2))
    backprop(a.splice(ind, target))
    grad_a_ = np.ones_like(a.data)
    grad_a_[2] = 0.
    grad_target = np.ones_like(target.data) * 2.
    assert a.grad is not None
    assert target.grad is not None
    assert np.array_equal(a.grad, grad_a_)
    assert np.array_equal(target.grad, grad_target)
    assert ind.grad is None
    reset_grads(a, target)

    # splice
    target = make_parameter(np.array(-4.))
    backprop(a.splice((slice(None, 2), slice(None, 2)), target))
    grad_a_ = np.ones_like(a.data)
    grad_a_[:2, :2] = 0.
    grad_b_ = np.ones_like(target.data) * 4.
    assert a.grad is not None
    assert target.grad is not None
    assert np.array_equal(a.grad, grad_a_)
    assert np.array_equal(target.grad, grad_b_)
    reset_grads(a, target)

    # splice
    target = make_parameter(np.array(-4.))
    ind = Tensor(np.array([True, False, True]))
    backprop(a.splice(ind, target))
    grad_a_ = np.ones_like(a.data)
    grad_a_[np.array([True, False, True])] = 0.
    grad_target = np.ones_like(target.data) * 4.
    assert a.grad is not None
    assert target.grad is not None
    assert np.array_equal(a.grad, grad_a_)
    assert np.array_equal(target.grad, grad_target)
    assert ind.grad is None
    reset_grads(a, target)

    # concatenate
    backprop(a.concat(b, dim=-1))
    assert a.grad is not None
    assert b.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a.data))
    assert np.array_equal(b.grad, np.ones_like(b.data))
    reset_grads(a, b)


def test_finite_difference(grad_error_tolerance=1e-4, dx=1e-6):
    dx = dx
    a = make_parameter(np.array(1.))
    b = make_parameter(np.array(0.))

    # add
    fn = Tensor.__add__
    args = (a, b)
    grads = finite_difference(dx, fn, *args)
    assert within_tolerance(grads[0], 1., grad_error_tolerance)
    assert within_tolerance(grads[1], 1., grad_error_tolerance)

    # exp
    def exp_fn(x, y):
        return x.exp() + y

    grads = finite_difference(dx, exp_fn, *args)
    assert within_tolerance(grads[0], np.e, grad_error_tolerance)
    assert within_tolerance(grads[1], 1., grad_error_tolerance)


# noinspection DuplicatedCode
def test_backprop(grad_error_tolerance=1e-4, dx=1e-6):
    # use float64, otherwise finite difference is wrong

    a = make_parameter(np.array([[1., 2.], [3., 4.], [5., 6.]]), dtype=float64)
    b = make_parameter(np.array([[3.], [2.], [1.]]), dtype=float64)

    # add
    def add(x, y):
        return (x + y).sum()

    with no_grad():
        fd_grads = finite_difference(dx, add, a, b)
    backprop(add(a, b))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    assert within_tolerance(fd_grads[1], b.grad, grad_error_tolerance)
    reset_grads(a, b)

    # sub
    def sub(x, y):
        return (x - y).sum()

    with no_grad():
        fd_grads = finite_difference(dx, sub, a, b)
    backprop(sub(a, b))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    assert within_tolerance(fd_grads[1], b.grad, grad_error_tolerance)
    reset_grads(a, b)

    # mult
    def mult(x, y):
        return (x * y).sum()

    with no_grad():
        fd_grads = finite_difference(dx, mult, a, b)
    backprop(mult(a, b))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    assert within_tolerance(fd_grads[1], b.grad, grad_error_tolerance)
    reset_grads(a, b)

    # true div
    def div(x, y):
        return (x / y).sum()

    with no_grad():
        fd_grads = finite_difference(dx, div, a, b)
    backprop(div(a, b))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    assert within_tolerance(fd_grads[1], b.grad, grad_error_tolerance)
    reset_grads(a, b)

    # transpose
    def transpose(x):
        return x.T.sum()

    with no_grad():
        fd_grads = finite_difference(dx, transpose, a)
    backprop(transpose(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # matmul
    def matmul(x, y):
        return (x.T @ y).sum()

    with no_grad():
        fd_grads = finite_difference(dx, matmul, a, b)
    backprop(matmul(a, b))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    assert within_tolerance(fd_grads[1], b.grad, grad_error_tolerance)
    reset_grads(a, b)

    # power
    def power(x, y):
        return (x ** y).sum()

    with no_grad():
        fd_grads = finite_difference(dx, power, a, b)
    backprop(power(a, b))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    assert within_tolerance(fd_grads[1], b.grad, grad_error_tolerance)
    reset_grads(a, b)

    # negative
    def negative(x):
        return (-x).sum()

    with no_grad():
        fd_grads = finite_difference(dx, negative, a)
    backprop(negative(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # exp
    def exp(x):
        return x.exp().sum()

    with no_grad():
        fd_grads = finite_difference(dx, exp, a)
    backprop(exp(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # log
    def log(x):
        return x.log().sum()

    with no_grad():
        fd_grads = finite_difference(dx, log, a)
    backprop(log(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # tanh
    def tanh(x):
        return x.tanh().sum()

    with no_grad():
        fd_grads = finite_difference(dx, tanh, a)
    backprop(tanh(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # sum
    def sum1(x):
        return x.sum()

    with no_grad():
        fd_grads = finite_difference(dx, sum1, a)
    backprop(sum1(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # sum
    def sum2(x):
        return x.sum(dim=0).sum()

    with no_grad():
        fd_grads = finite_difference(dx, sum2, a)
    backprop(sum2(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # sum
    def sum3(x):
        return x.sum(dim=1).sum()

    with no_grad():
        fd_grads = finite_difference(dx, sum3, a)
    backprop(sum3(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # mean
    def mean1(x):
        return x.mean().sum()

    with no_grad():
        fd_grads = finite_difference(dx, mean1, a)
    backprop(mean1(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # mean
    def mean2(x):
        return x.mean(dim=0).sum()

    with no_grad():
        fd_grads = finite_difference(dx, mean2, a)
    backprop(mean2(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # mean
    def mean3(x):
        return x.mean(dim=1).sum()

    with no_grad():
        fd_grads = finite_difference(dx, mean3, a)
    backprop(mean3(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # abs
    def abs(x):
        return x.abs().sum()

    with no_grad():
        fd_grads = finite_difference(dx, abs, a)
    backprop(abs(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # reshape
    def reshape(x):
        return x.reshape((-1,)).sum()

    with no_grad():
        fd_grads = finite_difference(dx, reshape, a)
    backprop(reshape(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # __getitem__
    def getitem1(x):
        return x[:2].sum()

    with no_grad():
        fd_grads = finite_difference(dx, getitem1, a)
    backprop(getitem1(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # __getitem__
    def getitem2(x):
        return x[:2, :2].sum()

    with no_grad():
        fd_grads = finite_difference(dx, getitem2, a)
    backprop(getitem2(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # __getitem__
    def getitem3(x):
        return x[Tensor(np.array([True, False, True]))].sum()

    with no_grad():
        fd_grads = finite_difference(dx, getitem3, a)
    backprop(getitem3(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # splice
    target = make_parameter(np.array(-4.))
    ind1 = Tensor(np.array(2))

    def splice1(x, key, val):
        return x.splice(key, val).sum()

    with no_grad():
        fd_grads = finite_difference(dx, splice1, a, ind1, target)
    backprop(splice1(a, ind1, target))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    assert fd_grads[1] is None and ind1.grad is None
    assert within_tolerance(fd_grads[2], target.grad, grad_error_tolerance)
    reset_grads(a, target)

    # splice
    ind2 = TensorSlice((slice(None, 2), slice(None, 2)))

    def splice2(x, key, val):
        return x.splice(key, val).sum()

    with no_grad():
        fd_grads = finite_difference(dx, splice2, a, ind2, target)
    backprop(splice2(a, ind2, target))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    assert fd_grads[1] is None and ind2.grad is None
    assert within_tolerance(fd_grads[2], target.grad, grad_error_tolerance)
    reset_grads(a, target)

    # splice
    ind3 = Tensor(np.array([True, False, True]))

    def splice3(x, key, val):
        return x.splice(key, val).sum()

    with no_grad():
        fd_grads = finite_difference(dx, splice3, a, ind3, target)
    backprop(splice3(a, ind3, target))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    assert fd_grads[1] is None and ind3.grad is None
    assert within_tolerance(fd_grads[2], target.grad, grad_error_tolerance)
    reset_grads(a, target)

    # concatenate
    def concat(x, y):
        return x.concat(y, dim=1).sum()

    with no_grad():
        fd_grads = finite_difference(dx, concat, a, b)
    backprop(concat(a, b))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    assert within_tolerance(fd_grads[1], b.grad, grad_error_tolerance)
    reset_grads(a, b)

    # matmul
    def matmul(x, y, z):
        return (x.T @ y + z).sum()

    z = make_parameter(np.array([[.54]]))
    with no_grad():
        fd_grads = finite_difference(dx, matmul, a, b, z)
    backprop(matmul(a, b, z))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    assert within_tolerance(fd_grads[1], b.grad, grad_error_tolerance)
    assert within_tolerance(fd_grads[2], z.grad, grad_error_tolerance)
    reset_grads(a, b)


def test_activations(grad_error_tolerance=1e-4, dx=1e-6):
    a = make_parameter(np.array([[-1., 0.5], [-0.2, 0.], [0.3, 100.]]), dtype=float64)

    # sigmoid
    def sigmoid_(x):
        return sigmoid(x).sum()

    with no_grad():
        fd_grads = finite_difference(dx, sigmoid_, a)
    backprop(sigmoid_(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # relu
    def rel(x):
        return relu(x).sum()

    with no_grad():
        # finite difference results in error around 0.
        fd_grads = (np.array([[0., 1.], [0., 0.], [1., 1.]]),)  # finite_difference(dx, fn, a)
    backprop(rel(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # log softmax
    def log_soft(x):
        return log_softmax(x).sum()

    with no_grad():
        # finite difference results in error around 0.
        fd_grads = finite_difference(dx, log_soft, a)
    backprop(log_soft(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)

    # cross entropy
    def ce(x):
        return cross_entropy_loss(x, Tensor(np.array([[0., 1.], [1., 0.], [0., 1.]]))).mean()

    with no_grad():
        # finite difference results in error around 0.
        fd_grads = finite_difference(dx, ce, a)
    backprop(ce(a))
    assert within_tolerance(fd_grads[0], a.grad, grad_error_tolerance)
    reset_grads(a)


# if __name__ == "__main__":
#     test_backward_ops()
#     test_finite_difference()
#     test_backprop()
#     test_activations()
