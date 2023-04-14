from __future__ import annotations

from contextlib import contextmanager
from typing import Union, Callable, Optional, Tuple, Any, Dict, Iterable

import numpy as np
import numpy.typing as npt

from backward_ops import backward_ops, np_splice_binary, np_concatenate_binary, np_mean, np_sum

float32 = np.dtype('float32')
float64 = np.dtype('float64')
int32 = np.dtype('int32')
int64 = np.dtype('int64')
bool_ = np.dtype('bool')
int_types = {int32, int64}
float_types = {float32, float64}
slice_types = int_types | {bool_}


index_np_array = Union[npt.NDArray[np.int32], npt.NDArray[np.int64], npt.NDArray[np.bool_]]
diff_np_array = Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]
tensor_np_array = Union[
    npt.NDArray[np.int32], npt.NDArray[np.int64], npt.NDArray[np.bool_],
    npt.NDArray[np.float32], npt.NDArray[np.float64]
]


_global_context = {
    "grad_enabled": True
}


def _get_context(key):
    return _global_context[key]


def _set_context(key, value):
    _global_context[key] = value


@contextmanager
def no_grad():
    """
    Context manager to disable gradient tracing globally.
    """
    prev_grad = _get_context("grad_enabled")
    _set_context("grad_enabled", False)
    try:
        yield None
    finally:
        _set_context("grad_enabled", prev_grad)


def _assert_tensor_args(args):
    assert all(isinstance(arg, _TensorPrimitive) for arg in args), \
        f"All `args` should have Tensor type, but " \
        f"received {tuple(type(arg) for arg in args)}, {tuple(isinstance(arg, _TensorPrimitive) for arg in args)}"


# noinspection PyProtectedMember
def _make_op_output_tensor(np_op, *args: _TensorPrimitive, **kwargs) -> Tensor:
    """
    Apply `np_op` to the current set of inputs
    :param np_op: Numpy function that should be applied
    :param args: Input arguments, should be inherited from _TensorPrimitive
    :param kwargs: Arbitrary arguments that are required by np_op but do not require gradient
    :return: New Tensor that contains the output of np_op. If gradients are enabled, stores information
        required for backpropagation
    """
    _assert_tensor_args(args)
    np_args = tuple(arg.data for arg in args)
    new_tensor = Tensor(np_op(*np_args, **kwargs))

    differentiable = False

    # Track gradients only if `grad_enabled` is set to true and the tensor has one of float types
    if _get_context("grad_enabled") and new_tensor.dtype in float_types:
        new_tensor._recipe = (args, kwargs)
        new_tensor._backward_op = backward_ops.get(np_op, None)
        differentiable = new_tensor._backward_op is not None

    new_tensor.requires_grad = differentiable
    return new_tensor


class _TensorPrimitive:
    """
    Base class for gradient computation and backpropagation. Used to store tensor data and gradient, and
    at the same time to track computational graph.
    """
    # Define some fields as private so that user is discouraged to interact with them
    _recipe: Optional[Tuple[Iterable[_TensorPrimitive], Dict[str, Any]]] = None  # inputs for creating this tensor
    _backward_op: Optional[Tuple[Callable, ...]] = None  # backpropagation ops for elements in `_recipe`

    data: Optional[Union[tensor_np_array, slice, Tuple]] = None  # placeholder for tensor data
    requires_grad: Optional[bool] = None  # whether gradients for this tensor should be computed
    grad: Optional[diff_np_array] = None  # placeholder for gradient value, populated during backpropagation

    def __init__(self, data_container):
        self.data = data_container

    @property
    def shape(self) -> Tuple:
        return ()

    @property
    def dtype(self):
        return None


class TensorSlice(_TensorPrimitive):
    """
    Used to wrap `slice` type. Needed for unifying DAG tracking and backpropagation
    """
    def __init__(self, data_container: Union[slice, Tuple[slice, ...]]):
        super().__init__(data_container)


class TensorShape(_TensorPrimitive):
    """
    Used to wrap shape tuple. Needed for unifying DAG tracking and backpropagation
    """
    def __init__(self, data_container: Tuple):
        super().__init__(data_container)


def make_parameter(value: Union[diff_np_array, Tensor], dtype=None, requires_grad=True) -> Tensor:
    """
    Helper function for creating trainable parameter. Initializes value from numpy array and sets
    `requires_grad` to `True`
    :param value: initial value
    :param dtype: data dtype
    :param requires_grad: Whether to compute gradient for created parameter
    :return: new tensor
    """
    if (
            (dtype is not None and dtype not in float_types) or
            (dtype is None and value.dtype not in float_types)
    ):
        raise TypeError("Need to use differentiable type")

    if isinstance(value, np.ndarray):
        value = Tensor(value.copy(), dtype=dtype)
    else:
        value = value.copy()

    value.requires_grad = requires_grad
    return value


def reset_grads(*args: Tensor):
    """
    Reset gradient values and `_recipe`
    :param args: tensors
    """
    for arg in args:
        arg.grad = None
        arg._backward_op = None
        arg._recipe = None


class Tensor(_TensorPrimitive):
    """
    Main tensor class used for mathematical operations. Inplace operations are not supported.
    """
    _backward_op: Optional[Tuple[Callable, ...]] = None
    data: tensor_np_array

    def __init__(self, value: tensor_np_array, dtype=None, requires_grad=False):
        if dtype is not None:
            _value = value.astype(dtype)
        else:
            _value = value

        if _value.dtype in float_types:
            self.requires_grad = requires_grad

        super().__init__(value)

    @staticmethod
    def _make_index_tensor(
            item: Union[TensorSlice, Tensor, index_np_array, slice, Tuple[slice, ...], int]
    ) -> Union[TensorSlice, Tensor]:
        item_: Union[Tensor, TensorSlice]
        if isinstance(item, slice) or isinstance(item, tuple) and all(isinstance(i, slice) for i in item):
            item_ = TensorSlice(item)
        elif isinstance(item, int):
            item_ = Tensor(np.array(item))
        elif isinstance(item, TensorSlice):
            item_ = item
        elif isinstance(item, np.ndarray) or isinstance(item, Tensor):
            if item.dtype not in slice_types:
                raise TypeError(f"Expecting one of the following types for slices: {slice_types}, but got {item.dtype}")
            if isinstance(item, np.ndarray):
                item_ = Tensor(item, dtype=item.dtype)
            else:
                item_ = item
        else:
            raise TypeError(f"Incorrect index type, expecting TensorSlice, Tensor, slice, Tuple[slice, ...],"
                            f" but received: {type(item)}")
        return item_

    @staticmethod
    def _make_other_tensor(other: Union[Tensor, int, float]) -> Tensor:
        if isinstance(other, int) or isinstance(other, float):
            return Tensor(np.array(other))
        return other

    @property
    def T(self) -> Tensor:
        return self.transpose()

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __getitem__(self, item: Union[Tensor, slice, Tuple[slice, ...]]) -> Tensor:
        return _make_op_output_tensor(np.ndarray.__getitem__, self, self._make_index_tensor(item))

    def __setitem__(self, key, value):
        raise NotImplementedError(
            "Since Tensor class is used to track computations inplace operations are not supported use splice instead"
        )

    def __neg__(self) -> Tensor:
        return _make_op_output_tensor(np.negative, self)

    def __add__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.add, self, self._make_other_tensor(other))

    def __radd__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.add, self, self._make_other_tensor(other))

    def __sub__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.subtract, self, self._make_other_tensor(other))

    def __rsub__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.subtract, self, self._make_other_tensor(other))

    def __mul__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.multiply, self, self._make_other_tensor(other))

    def __rmul__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.multiply, self, self._make_other_tensor(other))

    def __truediv__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.true_divide, self, self._make_other_tensor(other))

    def __rtruediv__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.true_divide, self._make_other_tensor(other), self)

    def __matmul__(self, other: Union[Tensor]) -> Tensor:
        return _make_op_output_tensor(np.matmul, self, other)

    def __pow__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.power, self, self._make_other_tensor(other))

    def __rpow__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.power, self._make_other_tensor(other), self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Tensor):
            return _make_op_output_tensor(np.equal, self, self._make_other_tensor(other)).all()
        raise NotImplementedError("Only comparison with other Tensors is implemented")

    def __lt__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.less, self, self._make_other_tensor(other))

    def __gt__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.greater, self, self._make_other_tensor(other))

    def __le__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.less_equal, self, self._make_other_tensor(other))

    def __ge__(self, other: Union[Tensor, int, float]) -> Tensor:
        return _make_op_output_tensor(np.greater_equal, self, self._make_other_tensor(other))

    def log(self) -> Tensor:
        return _make_op_output_tensor(np.log, self)

    def exp(self) -> Tensor:
        return _make_op_output_tensor(np.exp, self)

    def tanh(self) -> Tensor:
        return _make_op_output_tensor(np.tanh, self)

    def sum(self, dim=None) -> Tensor:
        return _make_op_output_tensor(np_sum, self, axis=dim)

    def mean(self, dim=None) -> Tensor:
        return _make_op_output_tensor(np_mean, self, axis=dim)

    def abs(self) -> Tensor:
        return _make_op_output_tensor(np.abs, self)

    def splice(self, key: Union[Tensor, slice, Tuple[slice, ...]], values: Union[Tensor, int, float]) -> Tensor:
        """
        Out of place replacement for __setitem__. See `np_splice_binary` for more details.
        """
        return _make_op_output_tensor(
            np_splice_binary, self, self._make_other_tensor(values), self._make_index_tensor(key)
        )

    def concat(self, other, dim=0) -> Tensor:
        return _make_op_output_tensor(np_concatenate_binary, self, other, axis=dim)

    def equal_elementwise(self, other: Tensor) -> Tensor:
        return _make_op_output_tensor(np.equal, self, other)

    def all(self) -> bool:
        return _make_op_output_tensor(np.all, self).data.item() is True

    def any(self) -> bool:
        return _make_op_output_tensor(np.any, self).data.item() is True

    def transpose(self) -> Tensor:
        return _make_op_output_tensor(np.transpose, self)

    def reshape(self, shape: Tuple) -> Tensor:
        if not isinstance(shape, tuple):
            raise TypeError("Parameter `shape` should be a tuple")
        shape_ = TensorShape(shape)
        return _make_op_output_tensor(np.reshape, self, shape_)

    def argmax(self, dim=-1) -> Tensor:
        assert dim is not None
        return _make_op_output_tensor(np.argmax, self, axis=-1)

    def __repr__(self) -> str:
        return repr(self.data).replace("array", "Tensor")

    def __hash__(self) -> int:
        return id(self)

    def copy(self) -> Tensor:
        return Tensor(self.data.copy())

    def item(self) -> Union[bool, int, float]:
        assert self.shape == ()
        return self.data.item()


unary_ops = {
    Tensor.__getitem__,
    Tensor.reshape,
    Tensor.transpose,
    Tensor.__neg__,
    Tensor.log,
    Tensor.exp,
    Tensor.tanh,
    Tensor.sum,
    Tensor.mean,
    Tensor.abs,
}


binary_ops = {
    Tensor.__add__,
    Tensor.__sub__,
    Tensor.__mul__,
    Tensor.__truediv__,
    Tensor.__matmul__,
    Tensor.__pow__,
    Tensor.concat
}


non_diff_unitary_ops = {
    Tensor.all,
    Tensor.any
}


non_diff_binary_unitary_ops = {
    Tensor.equal_elementwise,
    Tensor.__le__,
    Tensor.__lt__,
    Tensor.__ge__,
    Tensor.__gt__
}
