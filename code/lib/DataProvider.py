import os

import numpy as np

from config import DEFAULT_CONFIG

DEFAULT_SEED = 123456


class DataProvider:
    """Generic data provider with batching, shuffling, and iteration."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """
        Args:
            inputs: Array of data input features of shape (num_data, input_dim).
            targets: Array of data output targets.
            batch_size: Number of data points per batch.
            max_num_batches: Maximum batches per epoch (-1 for all data).
            shuffle_order: Whether to shuffle data before each epoch.
            rng: A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points per batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches per epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        return self

    def new_epoch(self):
        """Start a new epoch, optionally shuffling data."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def __next__(self):
        return self.next()

    def reset(self):
        """Reset the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffle data order."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Return the next data batch or raise StopIteration."""
        if self._curr_batch + 1 > self.num_batches:
            self.new_epoch()
            raise StopIteration()
        batch_slice = slice(
            self._curr_batch * self.batch_size,
            (self._curr_batch + 1) * self.batch_size,
        )
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch


class PascalDataProvider(DataProvider):
    """Data provider for brain tumor MRI images stored as .npz files."""

    def __init__(self, fileNumb, which_set='train', batch_size=100,
                 max_num_batches=-1, shuffle_order=True, rng=None):
        """
        Args:
            fileNumb: File number suffix for the .npz filenames.
            which_set: One of 'train', 'valid', or 'test'.
            batch_size: Number of data points per batch.
            max_num_batches: Maximum batches per epoch (-1 for all).
            shuffle_order: Whether to shuffle data before each epoch.
            rng: A seeded random number generator.
        """
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be train, valid, or test. Got {0}'.format(which_set)
        )
        self.which_set = which_set

        data_dir = DEFAULT_CONFIG.data_dir
        data_path = os.path.join(
            data_dir, '{0}{1}_input.npz'.format(which_set, fileNumb),
        )
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )

        loaded = np.load(data_path, allow_pickle=True)
        inputs = loaded['arr_0']

        target_path = os.path.join(
            data_dir, '{0}{1}_target.npz'.format(self.which_set, fileNumb),
        )
        targets = np.load(target_path, allow_pickle=True)['arr_0']

        super().__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng,
        )
