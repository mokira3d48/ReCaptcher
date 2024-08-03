import torch
from torch import nn


class Metric(object):

    def __init__(self, name=None):
        super().__init__()
        self._name = name if name else self.__class__.__name__
        self._sum = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        self._total = torch.tensor(0, dtype=torch.long, requires_grad=False)

    @property
    def name(self):
        """:str: the name of given to this metric instance"""
        return self._name

    def update_state(self, *args, **kwargs):
        """Method of updating state with a new value"""
        raise NotImplemented

    def result(self):
        # return torch.div(self._sum, self._total)
        return self._sum / self._total if self._total.item() != 0 \
            else self._sum

    def reset_state(self):
        # self._sum = torch.scalar(0, torch.float32, requires_grad=False)
        # self._total = torch.scalar(0, dtype=torch.long, requires_grad=False)
        self._sum.fill_(0)
        self._total.fill_(0)
