import numpy as np
import torch

from os import path
from typing import Tuple
from pathlib import Path


class ImageDataset:
    """
    Creates a dataset from numpy arrays while keeping the data
    in the more efficient numpy arrays for as long as possible and only
    converting to torchtensors when needed.
    """

    def __init__(self, x: Path, y: Path) -> None:
        # Target labels
        self.targets = ImageDataset.load_numpy_arr_from_npy(y)
        # Images
        self.imgs = ImageDataset.load_numpy_arr_from_npy(x)

    def __len__(self) -> int:
        return len(self.targets)

    # Here you can apply preprocessing to the data
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = torch.from_numpy(self.imgs[idx] / 255).float()
        label = self.targets[idx]
        return image, label

    @staticmethod
    def load_numpy_arr_from_npy(path: Path) -> np.ndarray:
        """
        Loads a numpy array from local storage.
        Input:
        path: local path of file
        Outputs:
        dataset: numpy array with input features or labels
        """

        return np.load(path)