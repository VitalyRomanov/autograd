# Autograd

Toy implementation of automatic gradients for training neural networks. See Jupyter notebook for quick introduction. 

- [x] Implement automatic differentiation
   - [x] Dynamic computational graph tracing
     - Class Tensor in [`tensor.py`](/tensor.py) serves for implementing tensor operations as well as tracking computational graph
     - Tensor overloads arithmetic (+, -, *, /, @) and logical elementwise operations (<, <=, >, >=, equal_elementwise)
     - Map operations (exp, log, pow, tanh, abs)
     - Reduce operations (sum, mean)
     - Manipulation (reshape, transpose, concatenate, __getitem__, splice (out-of-place __setitem__))
   - [x] Derivatives
     - Implementation is in [`backward_ops.py`](/backward_ops.py)
     - Derivatives for all arithmetical, map, reduce, manipulation operations
     - Derivatives for sigmoid and relu activations
   - [x] Backpropagation
     - Implementation of finite difference is in [`backprop.py`](/backprop.py), see `backprop` function
     - Context manager to disable gradient tracking during evaluation
   - [x] Test automatic gradients against gradients computed with finite differences
     - Implementation of finite difference is in [`finite_diff.py`](/finite_diff.py), see `finite_difference` function
     - Gradient test with a full neural network in Jupyter notebook
     - Gradient tests for basic operations are in [`test_grads.py`](/tests/test_grads.py).
- [x] NN training facilities
  - See [`modules.py`](/modules.py)
  - Has an implementation for a generic layer, Adam optimizer, and model trainer
  - Cross entropy loss can be found in [`activations_and_loss.py`](/activations_and_loss.py)
- [x] Static typing with MyPy
  - Provided type annotations for Tensor operators. MyPy issues an error when incorrect types are used, see Jypyter notebook
- [x] Train and test a model
  - Used `digits` dataset from sklearn
  - Data preprocessing and batching are implemented with sklearn
  - F1 score is averaged over batches