from collections import deque
from typing import Optional, Iterable, Deque

import numpy as np

from tensor import Tensor, _assert_tensor_args, _TensorPrimitive


# noinspection PyProtectedMember
def _order(start_node: _TensorPrimitive) -> Iterable[_TensorPrimitive]:
    """
    Iterate nodes in BFS order
    :param start_node: usually the input to `backprop` function
    :return: Iterator
    """
    queue: Deque[_TensorPrimitive] = deque()
    queue.append(start_node)

    while len(queue) > 0:
        node = queue.popleft()
        yield node
        if node._recipe is not None:
            parents, _ = node._recipe
            for parent in parents:
                queue.append(parent)


# noinspection PyProtectedMember
def backprop(root: Tensor):
    """
    :param root: Starting point for backpropagation
    :return: None
    """

    # need to state types explicitly for mypy
    fake_grad = np.ones_like(root.data, dtype=np.float32 if root.data.dtype == np.float32 else np.float64)
    root.grad = fake_grad
    for node in _order(root):
        if node.grad is None or node._backward_op is None or node._recipe is None:
            continue

        args, kwargs = node._recipe
        backward_ops = node._backward_op
        g = node.grad
        ans = node.data
        node.grad = None  # avoid reusing consumed gradient

        for parent, backward_op in zip(args, backward_ops):
            if parent.requires_grad and backward_op is not None:
                parent.grad = accumulate_grad(parent.grad, backward_op(g, ans, *(arg.data for arg in args), **kwargs))


# noinspection PyShadowingNames
def accumulate_grad(grad, value):
    """
    Accumulate gradient when tensor is reused several times during computation
    :param grad: Existing gradient value
    :param value: New gradient value
    :return: Accumulated gradient
    """
    if grad is None:
        return value
    else:
        return grad + value


def grad(fn, *args: Tensor, derivative_arg_ind: Optional[int] = None, **kwargs):
    """
    Compute gradient for a function using backpropagation
    :param fn: function under investigation
    :param args: input arguments for `fn`, should be tensors
    :param derivative_arg_ind:
    :param kwargs: arbitrary arguments that `fn` accepts. Gradients are not computed for them
    :return: Gradient value for each tensor in args
    """
    _assert_tensor_args(args)

    ans = fn(*args, **kwargs)
    backprop(ans)

    if derivative_arg_ind is None:
        return [arg.grad for arg in args]
    else:
        return args[derivative_arg_ind].grad
