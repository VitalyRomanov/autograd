import numpy as np

from modules import Module, ModuleTrainer
from tensor import make_parameter, Tensor


def test_model_trainer():
    class TestModel(Module):
        def __init__(self):
            super().__init__()
            self.weights = make_parameter(np.random.randn(20))

        def __call__(self, *args, **kwargs):
            return (self.weights ** Tensor(np.array(2))).sum()

        def loss(self, logits, y_true=None):
            return logits

    class TestModelTrainer(ModuleTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def make_model(self):
            self.model = TestModel()

    model_trainer = TestModelTrainer(learning_rate=1e-3, num_epochs=100)
    fake_batches = [(0, 0)] * 300
    model_trainer.fit(fake_batches)

    assert model_trainer.model.weights.sum().data.item() < 1e-20


# if __name__ == "__main__":
#     test_model_trainer()
