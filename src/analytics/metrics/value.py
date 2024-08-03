import torch
from .base import Metric


class Value(Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_value = torch.tensor(0)

    def update_state(self, value):
        self._current_value = torch.tensor(value)

    def result(self):
        return self._current_value

    def reset_state(self):
        pass
