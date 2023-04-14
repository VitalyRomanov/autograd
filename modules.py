from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from backprop import backprop
from tensor import Tensor, make_parameter, reset_grads, no_grad


def _is_module(var):
    return isinstance(var, Module)


class Module(ABC):
    def __init__(self):
        pass

    @property
    def parameters(self):
        for var in self.__dict__.values():
            if isinstance(var, Tensor) and var.requires_grad is True:
                yield var
            elif _is_module(var):
                yield from var.parameters

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...

    def loss(self, logits, y_true):
        return Tensor(np.array(0.))

    def score(self, logits, y_true):
        return Tensor(np.array(0.))


class Linear(Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        # initialization http://proceedings.mlr.press/v9/glorot10a.html
        gain_weights = np.sqrt(2 / (input_features + output_features))
        self.weights = make_parameter(np.random.normal(scale=gain_weights, size=(input_features, output_features)))
        if bias is True:
            self.bias = make_parameter(np.random.normal(size=(1, output_features)))

    def __call__(self, x: Tensor):
        """
        Forward pass
        :param x: input shape (None, input_features)
        :return:
        """
        assert len(x.shape) == 2, \
            "Only Tensors with dimension (None, num_features are supported)"
        assert x.shape[1] == self.weights.shape[0], \
            f"Dimension mismatch, expecting {self.weights.shape[0]} input features but received {x.shape[1]}"
        x = x @ self.weights
        if self.bias is not None:
            x = x + self.bias
        return x


class Optimizer(ABC):
    @abstractmethod
    def __init__(self, parameters, learning_rate, *args, **kwargs):
        ...

    @abstractmethod
    def step(self):
        ...


class AdamOptimizer(Optimizer):
    # https://arxiv.org/abs/1412.6980
    def __init__(
            self, parameters, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8
    ):
        self.parameters = list(set(parameters))  # make use parameters are unique
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.global_step = 1
        self._init_filter_memory()

    def _init_filter_memory(self):
        self._fm = {}
        self._sm = {}
        for parameter in self.parameters:
            self._fm[parameter] = np.zeros_like(parameter.data)
            self._sm[parameter] = np.zeros_like(parameter.data)

    def step(self):
        for p in self.parameters:
            self._fm[p] = self.beta_1 * self._fm[p] + (1 - self.beta_1) * p.grad
            self._sm[p] = self.beta_2 * self._sm[p] + (1 - self.beta_2) * p.grad ** 2
            m = self._fm[p] / (1 - self.beta_1 ** self.global_step)
            v = self._sm[p] / (1 - self.beta_2 ** self.global_step)
            p.data = p.data - self.learning_rate * m / (np.sqrt(v) + self.epsilon)

        self.global_step += 1


class ModuleTrainer(ABC):
    model: Any
    optimizer: Any

    def __init__(
            self, num_epochs, learning_rate
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.make_model()
        self.make_optimizer()

    @abstractmethod
    def make_model(self):
        # self.model = None
        ...

    def make_step(self, batch):
        x, y = batch
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        if isinstance(y, np.ndarray):
            y = Tensor(y)
        logits = self.model(x)
        loss = self.model.loss(logits, y)
        score = self.model.score(logits, y)
        return loss, score

    def make_optimizer(self):
        self.optimizer = AdamOptimizer(
            self.model.parameters, learning_rate=self.learning_rate,
        )

    def fit(self, train_data, test_data=None):

        for epoch in range(self.num_epochs):
            train_losses = []
            train_scores = []

            for batch in train_data:
                reset_grads(*self.optimizer.parameters)
                loss, score = self.make_step(batch)
                backprop(loss)
                self.optimizer.step()
                train_losses.append(loss.item())
                train_scores.append(score.item())

            train_avg_loss = sum(train_losses) / len(train_losses)
            train_avg_score = sum(train_scores) / len(train_scores)

            test_losses = []
            test_scores = []

            if test_data is not None:
                with no_grad():
                    for batch in test_data:
                        loss, score = self.make_step(batch)
                        test_losses.append(loss.item())
                        test_scores.append(score.item())

                test_avg_loss = sum(test_losses) / len(test_losses)
                test_avg_score = sum(test_scores) / len(test_scores)
            else:
                test_avg_loss = 0.
                test_avg_score = 0.

            print(f"Epoch {epoch}, Train Loss: {train_avg_loss:.4f}, Train F1 Score: {train_avg_score:.4f}, "
                  f"Test Loss: {test_avg_loss:.4f}, Test F1 Score: {test_avg_score:.4f}")

    def predict(self, x, y):
        ...
