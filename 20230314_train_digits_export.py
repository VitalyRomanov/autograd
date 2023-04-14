#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import gen_batches
from sklearn.metrics import f1_score

import numpy as np
from tensor import Tensor
from activations_and_loss import sigmoid, cross_entropy_loss
from modules import ModuleTrainer, Module, Linear
from matplotlib import pyplot as plt


# # Implementation details
# 
# Location of main components
# - Tensor definitions in `tensor.py`
# - Op gradients in `backward_ops.py`
# - Backpropagation in `backprop.py`
# - Finite difference gradient estimation in `finite_diff.py`
# - Trainable modules, optimizer, and model trainer class in `modules.py`
# - Activations and cross entropy loss in `activations_and_loss.py`

# ## Prepare data
# 
# Use digits dataset from sklearn, which contains 8x8 images of numbers. Preprocessing steps include
# 1. Split into train and test
# 2. Normalize features to range [0,1]
# 3. One hot encoding for labels
# 4. Preparing batches (no shuffling during training)

# In[2]:


data = load_digits()


# In[3]:


X_tr, X_te, Y_tr, Y_te = train_test_split(data["data"], data["target"], stratify=data["target"])


# In[4]:


X_tr = X_tr / X_tr.max()
X_te = X_te / X_tr.max()


# In[5]:


label_encoder = OneHotEncoder(sparse_output=False)
Y_tr_onehot = label_encoder.fit_transform(Y_tr.reshape((-1,1)))
Y_te_onehot = label_encoder.transform(Y_te.reshape((-1,1)))


# In[6]:


assert len(Y_tr_onehot) == len(X_tr)
assert len(Y_te_onehot) == len(X_te)
train_data = [
    (X_tr[slice_], Y_tr_onehot[slice_]) for slice_ in gen_batches(len(X_tr), batch_size=64)
]
test_data = [
    (X_te[slice_], Y_te_onehot[slice_]) for slice_ in gen_batches(len(X_te), batch_size=64)
]


# ## Define model and trainer
# 
# Define class `DigitsClassifier`
# 1. Three FC layers and sigmoid activation between layers
# 2. Hidden sizes are 40 and 20
# 3. Trained with cross entropy loss
# 
# Define `DigitsTrainer`
# 

# In[7]:


class DigitsClassifier(Module):
    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__()

        self.l1 = Linear(64, 40)
        self.l2 = Linear(40, 20)
        self.l3 = Linear(20, num_classes)

    def __call__(self, x):
        x = sigmoid(self.l1(x))
        x = sigmoid(self.l2(x))
        x = self.l3(x)
        return x

    def loss(self, logits, y_true):
        loss = cross_entropy_loss(logits, y_true)
        return loss.mean()

    def score(self, logits, y_true):
        y_pred = logits.argmax()
        y_true = y_true.argmax()
        return f1_score(y_true.data, y_pred.data, average="micro")

    def predict(self, x):
        logits = self(x)
        return logits.argmax()


# In[8]:


class DigitsTrainer(ModuleTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_model(self):
        self.model = DigitsClassifier()


# ## Training

# In[9]:


trainer = DigitsTrainer(num_epochs=300, learning_rate=1e-3)
trainer.fit(train_data=train_data, test_data=test_data)


# # Check results

# In[10]:


num_images = 5
test_images, test_labels = test_data[0][0][:num_images], test_data[0][1][:num_images]
for image, label in zip(test_images, test_labels):
    as_batch = Tensor(image.reshape(1,-1))
    pred = trainer.model.predict(as_batch)[0].item()
    print(f"Predicted: {pred}, Correct: {np.argmax(label)}")
    plt.imshow(image.reshape(8,8), interpolation='nearest')
    plt.show()


# ## Run pytest

# In[11]:


# get_ipython().system('pytest .')


# ## Run mypy

# In[12]:


# get_ipython().system('mypy activations_and_loss.py backprop.py backward_ops.py finite_diff.py modules.py tensor.py')


# ## Examples of catching incorrect types

# In[13]:


# get_ipython().system('mypy mypy_errors.py')


# ## Check gradients

# In[14]:


from backprop import grad as bp_grad
from finite_diff import finite_difference as fd_grad
from tensor import make_parameter, no_grad


def two_layers_nn(x, y, weights1, bias1, weights2, bias2):
    x = sigmoid(x @ weights1 + bias1)
    logits = x @ weights2 + bias2
    loss = cross_entropy_loss(logits, y)
    return loss.sum()

one_image = Tensor(np.reshape(train_data[0][0][0], (1, -1)))
one_label = Tensor(np.reshape(train_data[0][1][0], (1, -1)))
weights1 = make_parameter(np.random.normal(size=(one_image.shape[1], 20)))
bias1 = make_parameter(np.random.normal(size=(1, 20)))
weights2 = make_parameter(np.random.normal(size=(20, one_label.shape[1])))
bias2 = make_parameter(np.random.normal(size=(1, one_label.shape[1])))

with no_grad():
    fd_grads = fd_grad(1e-2, two_layers_nn, one_image, one_label, weights1, bias1, weights2, bias2)
bp_grads = bp_grad(two_layers_nn, one_image, one_label, weights1, bias1, weights2, bias2)


# In[15]:


difference_tolerance = 1e-5

assert len(fd_grads) == len(bp_grads)
for ind, (fd_grad, bp_grad) in enumerate(zip(fd_grads, bp_grads)):
    if fd_grad is None:
        assert bp_grad is None
        print(f"No gradients for argument {ind}")
        continue

    non_zero_mask = fd_grad != 0.
    fd_grad_ = fd_grad[non_zero_mask]
    bp_grad_ = bp_grad[non_zero_mask]

    diff = np.abs(fd_grad_ - bp_grad_) / np.abs(fd_grad_)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"For argument {ind} maximum diff is {max_diff*100:.5f}%, average diff {mean_diff*100:.5f}%")


# In[ ]:




