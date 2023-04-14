from itertools import product
from typing import Dict, Any

import numpy as np

from tensor import Tensor, unary_ops, no_grad, binary_ops, non_diff_unitary_ops, non_diff_binary_unitary_ops, \
    make_parameter


def iterate_grid(grid):
    keys = list(grid.keys())
    produced = False
    for values in product(*grid.values()):
        values = list(values)
        assert len(values) == len(keys)
        yield dict(zip(keys, values))
        produced = True

    if produced is False:
        yield {}


def test_ops():
    a = make_parameter(np.array([[1., 2.], [3., 4.], [-5., 6.]]))
    b = make_parameter(np.array([[3.], [2.], [1.]]))

    unary_extra_args = {
        Tensor.__getitem__: {"item": [slice(0, 2), (slice(0, 2), slice(0, 1)), np.array([True, True, False])]},
        Tensor.reshape: {"shape": [(-1, 1), (1, -1)]},
        Tensor.sum: {"dim": [None, -1, 0]},
        Tensor.mean: {"dim": [None, -1, 0]},
    }

    unary_ops_ans = {
        Tensor.__getitem__: {"ans": [Tensor(np.array([[1., 2.], [3., 4.]])), Tensor(np.array([[1.], [3.]])), Tensor(np.array([[1., 2.], [3., 4.]]))]},
        Tensor.reshape: {"ans": [Tensor(np.array([[1.], [2.], [3.], [4.], [-5.], [6.]])), Tensor(np.array([1., 2., 3., 4., -5., 6.]))]},
        Tensor.transpose: {"ans": [Tensor(np.array([[1., 3., -5.], [2., 4., 6.]]))]},
        Tensor.__neg__: {"ans": [Tensor(np.array([[-1., -2.], [-3., -4.], [5., -6.]]))]},
        Tensor.log: {"ans": [Tensor(np.log(a.data))]},
        Tensor.exp: {"ans": [Tensor(np.exp(a.data))]},
        Tensor.tanh: {"ans": [Tensor(np.tanh(a.data))]},
        Tensor.sum: {
            "ans": [Tensor(np.array(np.sum(a.data))), Tensor(np.array(np.sum(a.data, axis=-1))), Tensor(np.array(np.sum(a.data, axis=0)))]},
        Tensor.mean: {
            "ans": [Tensor(np.array(np.mean(a.data))), Tensor(np.array(np.mean(a.data, axis=-1))), Tensor(np.array(np.mean(a.data, axis=0)))]},
        Tensor.abs: {"ans": [Tensor(np.array([[1., 2.], [3., 4.], [5., 6.]]))]},
    }

    op: Any

    with no_grad():
        for op in unary_ops:
            kwargs = unary_extra_args.get(op, {})
            for grid, ans in zip(iterate_grid(kwargs), iterate_grid(unary_ops_ans[op])):
                # Cannot resolve op and falls back to the type of == output
                if op == Tensor.log:
                    assert ans["ans"].equal_elementwise(op(a, **grid)).sum().item() == 5
                else:
                    assert ans["ans"] == op(a, **grid)

    binary_extra_args: Dict[Any, Any] = {
        # Tensor.concat: {"dim": [1, -1]}
    }

    binary_ops_ans: Dict[Any, Any] = {
        Tensor.__add__: {"ans": [Tensor(a.data + b.data)]},
        Tensor.__sub__: {"ans": [Tensor(a.data - b.data)]},
        Tensor.__mul__: {"ans": [Tensor(a.data * b.data)]},
        Tensor.__truediv__: {"ans": [Tensor(a.data / b.data)]},
        # Tensor.__matmul__: {"ans": []},
        Tensor.__pow__: {"ans": [Tensor(a.data ** b.data)]},
        # Tensor.concat: {"ans": [Tensor(np.concatenate([a.data, b.data], axis=1)), Tensor(np.concatenate([a.data, b.data], axis=-1))]}
    }

    with no_grad():
        for op in binary_ops:
            if op in {Tensor.__matmul__, Tensor.concat}:
                continue
            kwargs = binary_extra_args.get(op, {})
            for grid, ans in zip(iterate_grid(kwargs), iterate_grid(binary_ops_ans[op])):
                # Cannot resolve op and falls back to the type of == output
                assert ans["ans"] == op(a, b, **grid)

    assert a.T @ b == Tensor(a.data.T @ b.data)

    c = Tensor(np.array([True, False, True]))
    non_diff_unitary_ops_ans = {
        Tensor.all: {"ans": [False]},
        Tensor.any: {"ans": [True]}
    }

    with no_grad():
        for op in non_diff_unitary_ops:
            for grid, ans in zip(iterate_grid(kwargs), iterate_grid(non_diff_unitary_ops_ans[op])):
                assert op(c) == ans["ans"]

    non_diff_binary_unitary_ops_ans = {
        Tensor.equal_elementwise: {"ans": [Tensor(a.data == b.data)]},
        Tensor.__le__: {"ans": [Tensor(a.data <= b.data)]},
        Tensor.__lt__: {"ans": [Tensor(a.data < b.data)]},
        Tensor.__ge__: {"ans": [Tensor(a.data >= b.data)]},
        Tensor.__gt__: {"ans": [Tensor(a.data > b.data)]}
    }

    with no_grad():
        for op in non_diff_binary_unitary_ops:
            for grid, ans in zip(iterate_grid(kwargs), iterate_grid(non_diff_binary_unitary_ops_ans[op])):
                assert op(a, b) == ans["ans"]

    # splice
    with no_grad():
        a_ = a.splice(Tensor(np.array(1)), -Tensor(np.array(4)))
        splice_ans = a.data.copy()
        splice_ans[1] = -4
        assert np.equal(a_.data, splice_ans).all()


# if __name__ == "__main__":
#     test_ops()
