# Autograd

Toy implementation of automatic gradients for training neural networks. See [Jupyter notebook](/20230314_train_digits.ipynb) for quick introduction. The file `20230314_train_digits_export.py` mirrors the content of the Jupyter notebook.

## Overview

Location of main components
- Tensor definitions are in `tensor.py`
- Op gradients are in `backward_ops.py`
- Backpropagation is in `backprop.py`
- Finite difference gradient estimation is in `finite_diff.py`
- Trainable modules, Adam optimizer, and model trainer classes are in `modules.py`
- Activations and cross entropy loss are in `activations_and_loss.py`

## Requirements

To run, need to install dependencies from `requirements.txt`.

## Automatic tests

```bash
bash run_pytest.sh
bash run_mypy.sh
bash run_mypy_example_errors.sh
```

## Implementation details

- [x] Implement automatic differentiation
   - [x] Dynamic computational graph tracing
     - Class Tensor in [`tensor.py`](/tensor.py) implements tensor operations as well as computational graph tracking
     - Tensor overloads arithmetic (+, -, *, /, @) and logical elementwise operations (<, <=, >, >=, equal_elementwise)
     - Implements some map operations (exp, log, pow, tanh, abs)
     - Implements some reduce operations (sum, mean)
     - Implements manipulation (reshape, transpose, etc.)
   - [x] Derivatives
     - See the implementation in [`backward_ops.py`](/backward_ops.py)
     - Derivatives for all arithmetical, map, reduce, manipulation operations
     - Derivatives for sigmoid and relu activations
   - [x] Backpropagation
     - See the implementation in [`backprop.py`](/backprop.py) (`backprop` function)
     - Context manager to disable gradient tracking during evaluation (`no_grad`)
   - [x] Test automatic gradients against gradients computed with finite differences
     - See the implementation of finite difference in [`finite_diff.py`](/finite_diff.py) (`finite_difference` function)
     - Gradient tests for a full neural network are in [Jupyter notebook](/20230314_train_digits.ipynb) (`Check gradients` section)
     - Gradient tests for basic operations are in [`test_grads.py`](/tests/test_grads.py).
- [x] NN training facilities
  - See [`modules.py`](/modules.py)
  - Has an implementation for a generic layer, Adam optimizer, and model trainer
  - Cross entropy loss can be found in [`activations_and_loss.py`](/activations_and_loss.py)
- [x] Static type checking with MyPy
  - Provided type annotations for Tensor operations. MyPy issues an error when incorrect types are used (see section `Examples of catching incorrect types` in [Jupyter notebook](/20230314_train_digits.ipynb))
- [x] Train and test a model
  - Used `digits` dataset from sklearn
  - Basic data preprocessing and batching are implemented with sklearn
  - F1 score is averaged over batches