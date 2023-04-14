from typing import Union, Tuple, Dict, Callable, Any

import numpy as np


def backwards_abs(g: np.ndarray, ans: np.ndarray, x: np.ndarray):
    """
    Gradient for `np.abs`. Equals 1 for x > 0., -1 for x < 0., and 0. for x == 0. (although gradient at 0.
    is not defined)
    :param g: previous gradient
    :param ans: output of `np.abs` operation
    :param x: input to `np.abs` operation
    :return: Gradient up to the current operation
    """
    negative_mask = x < 0.
    abs_grad = np.ones_like(x, dtype=x.dtype)
    abs_grad[negative_mask] = -1.
    abs_grad[x == 0.] = 0.
    return g * abs_grad


def backwards_slice(g: np.ndarray, ans: np.ndarray, l: np.ndarray, r: slice):
    """
    Gradient for `np.ndarray.__getitem__`. Equals 1. for retrieved items, 0. for the rest.
    :param g: previous gradient
    :param ans: output of `np.ndarray.__getitem__` operation
    :param l: input to `np.ndarray.__getitem__` operation
    :param r: key provided for `np.ndarray.__getitem__` operation
    :return: Gradient up to the current operation
    """
    slice_grad = np.zeros_like(l, dtype=l.dtype)
    slice_grad[r] = g
    return slice_grad


def np_concatenate_binary(l: np.ndarray, r: np.ndarray, axis: int = 0):
    """
    Concatenate two numpy arrays
    :param l: left operand
    :param r: right operand
    :param axis: axis along which to concatenate
    :return: concatenated numpy array
    """
    return np.concatenate([l, r], axis=axis)


def np_splice_binary(l: np.ndarray, r: np.ndarray, mask: Union[np.ndarray, slice, Tuple[slice, ...]]):
    """
    Function used as out-of-place replacement for `np.ndarray.__setitem__`.  Creates a new array from `l` where elements
    specified with `mask` are replaced with `r`
    :param l: Source data
    :param mask: Which elements to replace
    :param r: Which elements t replace with
    :return: Copy of array `l` with part of its elements replaced by `r`
    """
    l = l.copy()
    l[mask] = r
    return l


def backwards_np_splice_binary_l(g: np.ndarray, ans: np.ndarray, l: np.ndarray, r: np.ndarray, mask: np.ndarray):
    """
    Backwards operation for the left operand of np_splice_binary
    :param g: Previous gradient
    :param ans: Output of np_splice_binary operation
    :param l: Source data
    :param mask: Which elements to replace
    :param r: Which elements t replace with
    :return: Gradients for the left operand
    """
    grad = np.ones_like(l)
    grad[mask] = 0.
    return grad


def backwards_np_splice_binary_r(g: np.ndarray, ans: np.ndarray, l: np.ndarray, r: np.ndarray, mask: np.ndarray):
    """
    Backwards operation for the right operand of np_splice_binary
    :param g: Previous gradient
    :param ans: Output of np_splice_binary operation
    :param l: Source data
    :param mask: Which elements to replace
    :param r: Which elements t replace with
    :return: Gradients for the right operand
    """
    return unbroadcast(g[mask], r)


def np_sigmoid(x: np.ndarray):
    return 1. / (1. + np.exp(-x))


def np_relu(x: np.ndarray):
    x = x.copy()
    x[x <= 0] = 0.
    return x


def backwards_np_relu(g: np.ndarray, ans: np.ndarray, x: np.ndarray):
    grad = np.ones_like(x)
    grad[x <= 0] = 0.
    return grad * g


def np_sum(x, axis=None, keepdims=False):
    """
    Replacement for `np.sum` to conform with type requirements. Cast the output of `np.sum` to `np.ndarray`.
    """
    return np.array(np.sum(x, axis=axis, keepdims=keepdims))


def np_mean(x, axis=None, keepdims=False):
    """
    Replacement for `np.mean` to conform with type requirements. Cast the output of `np.mean` to `np.ndarray`.
    """
    return np.array(np.mean(x, axis=axis, keepdims=keepdims))


def unbroadcast(grad: np.ndarray, arg: np.ndarray):
    grad_s = grad.shape
    arg_s = arg.shape

    if arg_s == ():  # when arg is a scalar
        arg_s = (1,) * len(grad_s)

    if len(grad_s) != len(arg_s):
        raise Exception(f"Broadcasting dimension size mismatch: {grad_s}, {arg_s}")

    for axis, (g, a) in enumerate(zip(grad_s, arg_s)):
        if g != a:
            # assert min(g, a) == 1
            grad = np.sum(grad, axis=axis, keepdims=True)

    if arg.shape == ():  # if it was a scalar, make it a again a scalar
        grad = np.squeeze(grad)

    return grad


def unsqueeze(array, dim):
    if dim is None:
        return array
    else:
        return np.expand_dims(array, dim)


def axis_size(array, axis):
    if axis is None:
        return array.size
    else:
        return array.shape[axis]


backward_ops: Dict[Any, Tuple[Callable, ...]] = {
    np.add: ((lambda g, ans, l, r: unbroadcast(g, l)), (lambda g, ans, l, r: unbroadcast(g, r))),
    np.subtract: ((lambda g, ans, l, r: unbroadcast(g, l)), (lambda g, ans, l, r: unbroadcast(-g, r))),
    np.multiply: ((lambda g, ans, l, r: unbroadcast(r * g, l)), (lambda g, ans, l, r: unbroadcast(l * g, r))),
    np.true_divide: (
        (lambda g, ans, l, r: unbroadcast(g / r, l)),
        (lambda g, ans, l, r: unbroadcast(- g * l / r**2, r))
    ),
    np.matmul: ((lambda g, ans, l, r: unbroadcast(g @ r.T, l)), (lambda g, ans, l, r: unbroadcast(l.T @ g, r))),
    np.power: (
        (lambda g, ans, l, r: unbroadcast(g * r * ans / l, l)),
        (lambda g, ans, l, r: unbroadcast(g * ans * np.log(l), r))
    ),
    np.negative: (lambda g, ans, x: -g,),
    np.log: (lambda g, ans, x: g / x,),
    np.exp: (lambda g, ans, x: ans * g,),
    np.tanh: (lambda g, ans, x: g * (1 - ans ** 2),),  # g / np.cosh(x) ** 2,),  # g * (1 - ans ** 2)
    np_sum: (lambda g, ans, x, axis: unsqueeze(g, axis) * np.ones_like(x, dtype=x.dtype),),
    np_mean: ((lambda g, ans, x, axis: unsqueeze(g, axis) * np.ones_like(x, dtype=x.dtype) / axis_size(x, axis)),),
    np.abs: (backwards_abs,),

    np.transpose: (lambda g, ans, x: np.transpose(g),),
    np.ndarray.__getitem__: (backwards_slice,),
    np.reshape: (lambda g, ans, l, r: g.reshape(l.shape),),
    np_concatenate_binary: (
        lambda g, ans, l, r, axis: np.take(g, indices=range(0, l.shape[axis]), axis=axis),
        lambda g, ans, l, r, axis: np.take(g, indices=range(l.shape[axis], g.shape[axis]), axis=axis),
    ),
    np_splice_binary: (backwards_np_splice_binary_l, backwards_np_splice_binary_r),
    np_sigmoid: (lambda g, ans, x: g * ans * (1 - ans),),
    np_relu: (backwards_np_relu,),
}
