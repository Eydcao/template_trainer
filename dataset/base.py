import os
import glob
import numpy as np
import torch
from torch.utils.data import IterableDataset


class BaseDataPipe(IterableDataset):
    """
    Base class for data pipeline, implementing an iterable dataset.

    This class handles data loading, processing, and iteration for train, test, and rollout modes.
    It's designed to be subclassed for specific dataset implementations.
    """

    def __init__(self, cfg, num_workers, base_seed, mode):
        """
        Initialize the BaseDataPipe object.

        Args:
            cfg (OmegaConf): Configuration object containing all necessary parameters.
            num_workers (int): Number of worker processes for data loading.
            base_seed (int): Base seed for random number generators.
            mode (str): Mode of the dataset, either 'train', 'test', or 'rollout'.
        """
        self.cfg = cfg
        self.num_workers = num_workers
        self.base_seed = base_seed
        self.train = mode == "train"
        self.rollout = mode == "rollout"
        # For rollout mode, we use the test data directory
        self.mode = "test" if self.rollout else mode
        self.data_dir = os.path.join(cfg.dataset_root, self.mode)
        self.file_list = self._get_file_list()
        self.rng = None
        self.tc_rng = None
        self._init_rng()

    # =====================================================================
    # Methods that may need to be implemented in child classes
    # =====================================================================

    def _get_file_list(self):
        """
        Get the list of files in the data directory.

        Returns:
            list: List of file paths.
        """
        return glob.glob(os.path.join(self.data_dir, "*.h5"))

    def _read_path(self, file_path):
        """
        Read the trajectory data from the file.

        This method should be implemented in child classes.

        Args:
            file_path (str): Path to the file.

        Raises:
            NotImplementedError: If not implemented in the child class.
        """
        raise NotImplementedError("_read_path should be implemented in the child class.")

    def _get_slice(self, data, index):
        """
        Get a slice of the data for a specific batch and time step.
        Handles both single tensor and tuple data formats.

        Args:
            data: Single tensor or tuple of tensors
            index (int): Index of the slice to retrieve

        Returns:
            Single tensor slice or tuple of slices
        """
        if isinstance(data, tuple):
            return tuple(d[index] for d in data)
        else:
            return data[index]

    def _proc_data(self, data, rng, tc_rng):
        """
        Process the data and optionally inject noise.

        In this implementation, no additional processing is done.

        Args:
            data (tuple): Tuple of (input features, target features).
            rng (np.random.Generator): Random number generator for numpy.
            tc_rng (torch.Generator): Random number generator for torch.

        Returns:
            tuple: Processed (input features, target features).
        """
        return data

    # =====================================================================
    # Methods that do not need modifications in child classes
    # =====================================================================

    def _init_rng(self):
        """Initialize different random generators for each worker."""
        worker_id, _ = self._get_worker_id_and_info()
        train_seed = np.random.randint(1000) if self.train else np.random.randint(2000)
        seed_list = [train_seed, worker_id, self.base_seed]
        seed = self._hash(seed_list, 1000)
        self.rng = np.random.default_rng(seed)
        self.tc_rng = torch.Generator()
        self.tc_rng.manual_seed(seed)

    def _get_worker_id_and_info(self):
        """Get worker ID and info for multi-process data loading."""
        worker_info = torch.utils.data.get_worker_info()
        return (0, None) if worker_info is None else (worker_info.id, worker_info)

    def _hash(self, list, base):
        """
        Compute a hash for a list using a base value.

        Args:
            list (list): List to hash.
            base (int): Base value for hashing.

        Returns:
            int: Computed hash value.
        """
        hash = 0
        for i, value in enumerate(list):
            hash += value * (base**i)
        return hash

    def _get_nested_paths(self):
        """
        Split data paths based on the number of workers.

        Returns:
            list: List of nested paths for each worker.
        """
        if self.num_workers <= 1:
            return [self.file_list]
        elif self.num_workers > len(self.file_list):
            return [self.file_list for _ in range(self.num_workers)]
        else:
            return np.array_split(self.file_list, self.num_workers)

    def __iter__(self):
        """
        Iterate over the dataset, yielding data samples.

        Yields:
            tuple: Tuple of input features and target features.
        """
        worker_id, _ = self._get_worker_id_and_info()
        worker_paths = self._get_nested_paths()[worker_id]
        if self.rng is not None:
            self.rng.shuffle(worker_paths)

        for file_path in worker_paths:
            data, length = self._read_path(file_path)
            b_ids = np.arange(length)

            if self.rng is not None:
                self.rng.shuffle(b_ids)

            for bi in b_ids:
                sample = self._get_slice(data, bi)
                proc_data = self._proc_data(sample, self.rng, self.tc_rng)
                yield proc_data
