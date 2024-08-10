import yaml
import torch
from torch import nn
from torchinfo import summary


class Input(nn.Module):
    """Definition of input module"""

    def __init__(self, shape, dtype=torch.float32):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def get_zeros_tensor(self, batch_size=1):
        """
        Method to return a batch of zeros tensor
        with same shape and dtype as this input.
        """
        return torch.zeros([batch_size, *self.shape], dtype=self.dtype)

    def forward(self, x):
        """Forward method definition"""
        assert x.shape == self.shape, (
            "The shape of x value is not equal to input shape"
        )
        assert x.dtype == self.dtype, (
            f"{self.dtype} expected, but we received {x.dtype}"
        )
        return x


class Model(nn.Module):
    """Definition of customized model"""

    def __init__(self, inputs=None):
        super().__init__()
        self.inputs = [inputs] if not hasattr(inputs, '__iter__') \
            else inputs

    @classmethod
    def load_config(cls, **kwargs):
        raise NotImplemented("You should implement this method.")

    @classmethod
    def from_config_file(cls, file_path):
        """Method of model config loading from yaml file"""
        args = {}
        with open(file=file_path, mode='r') as file:
            args = yaml.load(file, Loader=yaml.FullLoader)

        instance = cls.load_config(**args)
        return instance

    def summary(self, batch_size=1):
        """Method of model summarization"""
        if not self.inputs:
            raise ValueError(
                "No `Input` model passed in constructor."
                " Please create an analytics.models.Input instance"
                f" and pass it in argument to this model: {self.__class__}."
            )

        input_data = [inp.get_zeros_tensor(batch_size) for inp in self.inputs]
        summary(self, input_data=input_data)
