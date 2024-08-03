import logging
import torch
from torch import nn
from .pbar import Progress


LOG = logging.getLogger(__name__)


class BaseInference(nn.Module):
    """Base inference implementation"""

    def __init__(
        self,
        device = None,
        pbar_format = None,
        log_format = None,
    ):
        super().__init__()
        self._device = device
        self._pbar_format = pbar_format # if pbar_format \
            # else ("{logger}"
            #       "[\033[92m{purcent:6.2f}\033[0m]"
            #       "{pbar}"
            #       "- [{time_rem} - {it_rate} its/s]"
            #     )
        self._log_format = log_format
        if not device:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def compile(self, *args, **kwargs):
        """Training compilation function

        This function is used to configure
        the inference algorithm with the model.
        """
        raise NotImplemented

    def on_start(self):
        """Method executed on inference starting

        In this method, you can call `model.train()`.
        """
        pass

    def step(self, x):
        """Inference method of the model

        It's use to predict one batch of the data.
        """
        raise NotImplemented

    def update_result(self, results, result):
        """Method of updating result

        :param results: The list of result for each prediction.
        :param result: The result of the current prediction.

        :type results: Dict[str, List[Any]]
        :type result: Dict[str, Any]
        :rtype: Dict[str, List[Any]]
        """
        for attrname, value in result.items():
            if attrname in results:
                results[attrname].append(value)
            else:
                results[attrname] = [value]

        return results  # we return updated results;

    def forward(self, input_x):
        """Inference running method"""
        results = {}
        pbar = Progress(len(input_x),
                        bins=15,
                        barf=self._pbar_format,
                        log_format=self._log_format,)

        outputs = []
        pbar.log('message', 'infer...')
        pbar.log()
        self.on_start()
        for x in input_x:
            result = {}
            with torch.no_grad():
                returned = self.step(x)

            if isinstance(returned, tuple):
                outputs.append(returned[0])
                if len(returned) > 1:
                    result = returned[1]
            else:
                outputs.append(returned)

            pbar.step(1)
            if not isinstance(result, dict):
                continue

            pbar.loginfo.update(result)
            pbar.log()
            self.update_result(results, result)

        pbar.log('message', 'DONE!')
        pbar.finalise()
        pbar.reset()

        # outputs = torch.tensor(outputs)
        return outputs, results
