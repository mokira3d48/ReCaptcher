# fig, axs = plt.subplots(1, 2)
# axs[0].plot([1, 2, 3], [0, 0.5, 0.2])
# axs[1].plot([3, 2, 1], [0, 0.5, 0.2])
import os
import matplotlib.pyplot as plt
from .callbacks import Callback


class TrainRepport(Callback):
    """Training repporting programming"""

    def __init__(self,
                 trainer,
                 outputs_dir = 'runs/',
                 train_dir = 'training',
                 valid_dir = 'validation',
                 figsize = (12, 8),
                 *args, **kwargs):

        super().__init__(trainer)
        self._outputs_dir = outputs_dir
        self._train_dir = train_dir
        self._valid_dir = valid_dir
        self._fig_size = figsize

        self._train_dp = os.path.join(outputs_dir, train_dir)
        self._valid_dp = os.path.join(outputs_dir, train_dir)

        if not os.path.isdir(self._train_dp):
            os.makedirs(self._train_dp)

        if not os.path.isdir(self._valid_dp):
            os.makedirs(self._valid_dp)


    @property
    def train_results(self):
        return self.trainer.train_results

    @property
    def valid_results(self):
        return self.trainer.valid_results

    def on_epoch_end(self):
        # result_names = []
        # result_names.extends(list(self.train_results.keys()))
        # result_names.extends(list(self.valid_results.keys()))
        for name, train_values in self.train_results.items():
            epochs = list(range(1, len(train_values) + 1))
            fn_result = os.path.join(self._train_dp, f"{name}_result.png")
            plt.figure(figsize=self._fig_size)
            plt.plot(epochs, train_values, label="Training")
            if name in self.valid_results:
                valid_values = self.valid_results[name]
                epochs = list(range(1, len(valid_values) + 1))
                plt.plot(epochs, valid_values, label="Validation")

            plt.title(name)
            plt.xlabel("Epochs")
            plt.ylabel(f"Value of {name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(fn_result)
            plt.clf()
            plt.close()

        for name, values in self.valid_results.items():
            if name not in self.train_results:
                epochs = list(range(1, len(values) + 1))
                fn_result = os.path.join(self._train_dp, f"{name}_result.png")
                plt.figure(figsize=self._fig_size)
                plt.plot(epochs, values, label="Valudation")
                plt.title(f"{name} of validation")
                plt.xlabel("Epochs")
                plt.ylabel(f"Value of {name}")
                plt.legend()
                plt.grid(True)
                plt.savefig(fn_result)
                plt.clf()
                plt.close()
