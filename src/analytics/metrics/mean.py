import torch
from .base import Metric


class Mean(Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, value):
        self._sum = torch.add(self._sum, value)
        self._total = torch.add(self._total, 1)
