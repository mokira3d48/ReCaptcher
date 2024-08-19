import gc

from sympy import preview
# from torch import nn
from torch.utils.data import (DataLoader as PyTorchDataLoader,
                              RandomSampler,
                              SubsetRandomSampler)


class DataLoader:

    def __init__(
            self,
            dataset,
            batch_size=32,
            shuffle=False,
            initial_batch_indices=None,
            num_workers=1,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_worker = num_workers

        self._pytorch_dl = None  # data loader instance of pytorch.
        self._data_iter = None
        self._sampler = None
        self._batch_index = None
        self.init_it(
            batch_indices=initial_batch_indices,
        )

    @property
    def batch_index(self):
        """int: returns the current value of batch index"""
        return self._batch_index

    @property
    def batch_indices(self):
        """list of int: returns the list of batch indices"""
        if self._sampler:
            return list(self._sampler)

    def _free_memory(self):
        if self._pytorch_dl:
            del self._pytorch_dl

        if self._sampler:
            del self._sampler
            self._sampler = None

        gc.collect()

    def _init_random_sampler(self, batch_indices=None):
        if batch_indices is not None:
            self._sampler = batch_indices
        elif self.shuffle:
            sampler = RandomSampler(self.dataset, replacement=False)
            self._sampler = list(sampler)
        else:
            self._sampler = list(range(len(self.dataset)))

    def _init_data_loader(self):
        self._pytorch_dl = PyTorchDataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            shuffle=False,
            sampler=self._sampler,
        )
        self._data_iter = iter(self._pytorch_dl)
        return self._pytorch_dl

    def init_it(self, batch_indices=None, batch_index=None):
        if batch_index is not None:
            self._batch_index = batch_index

        if self._batch_index is None:
            self._batch_index = 0

        self._free_memory()
        self._init_random_sampler(batch_indices)
        self._init_data_loader()

    def reset_it(self):
        self._batch_index = 0
        self._free_memory()
        self._init_random_sampler()
        self._init_data_loader()

    def _move_to(self, index, start=0):
        counter = start
        limit = index - 1
        while counter <= limit:
            next(self._data_iter)
            counter += 1

    def set_batch_index(self, selected_index):
        """Method to set the value of the current batch index"""
        if selected_index is None:
            selected_index = 0

        if selected_index < 0 or selected_index > len(self._pytorch_dl):
            raise IndexError()

        if selected_index == self._batch_index:
            return

        if selected_index < self._batch_index:
            batch_indices = self.batch_indices
            self.init_it(batch_indices=batch_indices, batch_index=0)

        self._move_to(selected_index, start=self._batch_index)
        self._batch_index = selected_index

    def __len__(self):
        return len(self._pytorch_dl)

    def __iter__(self):
        return self

    def __next__(self):
        if self._batch_index < len(self):
            returned = next(self._data_iter)
            self._batch_index += 1
            return returned
        else:
            self.reset_it()
            raise StopIteration("End of data iteration!")


def main():
    import torch
    from torch.utils.data import TensorDataset

    # Creation of dataset
    data = torch.randn(100, 10)  # 100 samples
    targets = torch.randint(0, 2, (100,))  # 100 binary labels
    dataset = TensorDataset(data, targets)

    data_loader = DataLoader(dataset,
                             shuffle=False,
                             initial_batch_indices=list(range(100)))
    data_loader.set_batch_index(3)
    print(data_loader.batch_indices)
    for x, y in data_loader:
        print(x.shape, y[:8])

    print("-------------------------------")
    print(data_loader.batch_indices)
    data_loader.set_batch_index(4)
    print(data_loader.batch_indices)
    data_loader.set_batch_index(1)
    for x, y in data_loader:
        print(x.shape, y[:8])

    print("-------------------------------")
    for x, y in data_loader:
        print(x.shape, y[:8])


if __name__ == '__main__':
    main()
