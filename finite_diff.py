from typing import List, Optional, Callable

import numpy as np

from backprop import _assert_tensor_args
from tensor import Tensor, float64, float_types, no_grad, _TensorPrimitive


def _increment_linear_position(array: Tensor, linear_ind: int, value: float):
    """
    Create a copy of `array` and add `value` to the position in the tensor specified with `linear_ind`
    :param array: tensor where the value is to be changed
    :param linear_ind: Position of the value in a flattened `array` to change
    :param value: Value that is to be added
    :return: Copy of `array` with vne value at position `linear_ind` modified
    """
    orig_shape = array.shape
    array = array.copy()
    array = array.reshape((-1,))
    array.data[linear_ind] += value
    return array.reshape(orig_shape)


def _swap_arg(args, ind, value):
    """
    Replace one of the values in `args` with `value`, original `args` preserved
    :param args: arguments
    :param ind: position that should be replaced
    :param value: new value
    :return: modified `args`
    """
    args = list(args)
    args[ind] = value
    return args


def _compute_diff(
        fn, args_u: List[Tensor], args_l: List[Tensor], **kwargs
):
    """
    Compute difference between the output of the function `fn` using arguments `args_u` and `args_l`
    :param fn: function under investigation
    :param args_u: set of arguments for LHS of the difference
    :param args_l: set of arguments for RHS of the difference
    :param kwargs: arbitrary kwargs that `fn` expects
    :return: The difference between two outputs.
    """
    diff = (fn(*args_u, **kwargs) - fn(*args_l, **kwargs))
    assert diff.shape == (), "Finite difference gradient can be computed only for functions that return scalar value"
    return diff.item()


def _assert_finite_diff_dtype(arg):
    assert arg.dtype == float64, "Data type should be float64, otherwise gradient errors are too large"


def _finite_difference_scalar(
        dx: float, fn: Callable, *args: _TensorPrimitive, derivative_arg_ind: int, **kwargs
):
    """
    Compute finite difference gradient estimate for a scalar argument
    :param dx: argument increment for computing the gradient
    :param fn: function under investigation
    :param args: list of arguments for the current function
    :param derivative_arg_ind: position of the scalar argument, for which the gradient is computed
    :param kwargs: arbitrary kwargs that `fn` expects
    :return: Gradient estimate for the argument `args[derivative_arg_ind]`
    """
    diff_arg = args[derivative_arg_ind]

    assert diff_arg.shape == (), "Non scalar argument for scalar differentiation"

    if not isinstance(diff_arg, Tensor) or diff_arg.dtype not in float_types:
        # no differentiation for non-float types
        return None

    _assert_finite_diff_dtype(diff_arg)

    args_u = _swap_arg(args, derivative_arg_ind, diff_arg + dx)
    args_l = _swap_arg(args, derivative_arg_ind, diff_arg - dx)

    return _compute_diff(fn, args_u, args_l, **kwargs) / (2. * dx)


def _finite_difference_tensor(
        dx: float, fn: Callable, *args: _TensorPrimitive, derivative_arg_ind: int, **kwargs
):
    """
    Compute finite difference gradient estimate for a tensor argument
    :param dx: argument increment for computing the gradient
    :param fn: function under investigation
    :param args: list of arguments for the current function
    :param derivative_arg_ind: position of the tensor argument, for which the gradient is computed
    :param kwargs: arbitrary kwargs that `fn` expects
    :return: Gradient estimate for the argument `args[derivative_arg_ind]`
    """
    diff_arg = args[derivative_arg_ind]

    assert len(diff_arg.shape) > 0, "Scalar argument for tensor differentiation"

    if not isinstance(diff_arg, Tensor) or diff_arg.dtype not in float_types:
        # no differentiation for non-float types
        return None

    _assert_finite_diff_dtype(diff_arg)

    grads_ = []
    with no_grad():
        for i in range(diff_arg.size):
            # compute gradient estimate for each position in the tensor
            args_u = _swap_arg(args, derivative_arg_ind, _increment_linear_position(diff_arg, i, dx))
            args_l = _swap_arg(args, derivative_arg_ind, _increment_linear_position(diff_arg, i, -dx))

            diff = _compute_diff(fn, args_u, args_l, **kwargs) / (2. * dx)
            grads_.append(diff)

    grads = np.array(grads_, dtype=float64).reshape(diff_arg.shape)
    return grads


def finite_difference(
        dx: float, fn, *args: _TensorPrimitive, derivative_arg_ind: Optional[int] = None, **kwargs
):
    """
    Handles several cases:
    1. If `derivative_arg_ind` is set to None then compute gradients for all `args`
    2. If `derivative_arg_ind` is an int then compute gradient only for that argument
        - If args[derivative_arg_ind] is a scalar Tensor (shape == ()), compute scalar gradient
        - If args[derivative_arg_ind] is a Tensor, compute vjp
    :param dx: argument increment
    :param fn: function to estimate derivative, should take `args` and `kwargs` as arguments
    :param args: All `args` are expected to be Tensors
    :param derivative_arg_ind: None or int in {0, len(args)}
    :param kwargs: kwargs are passed to fn, gradients are not computed for them, can be of arbitrary type
    :return: if derivative_arg_ind is int returns gradient for args[derivative_arg_ind], if derivative_arg_ind is None
        returns gradients for all Tensors in args
    """
    _assert_tensor_args(args)

    if derivative_arg_ind is None:
        # when derivative_arg_ind is None compute derivative with respect to all inputs
        return tuple(
            finite_difference(dx, fn, *args, derivative_arg_ind=i, **kwargs)
            for i in range(len(args))
        )
    else:
        diff_arg = args[derivative_arg_ind]
        if diff_arg.requires_grad is False:
            return None

        if diff_arg.shape == ():
            return _finite_difference_scalar(dx, fn, *args, derivative_arg_ind=derivative_arg_ind, **kwargs)

        return _finite_difference_tensor(dx, fn, *args, derivative_arg_ind=derivative_arg_ind, **kwargs)
