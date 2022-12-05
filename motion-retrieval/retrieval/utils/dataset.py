from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset


class MotionDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return len(self.data["observations"])

    def __getitem__(self, idx):
        return (
            self.data["observations"][idx],
            self.data["future_observations"][idx],
        )


class MotionDataLoader(DataLoader):
    def __init__(
        self,
        data_path: str = None,
        data: Dict[str, torch.Tensor] = None,
        **kwargs,
    ):
        assert (data_path or data) is not None, "Must provide either data_path or data"
        assert (data_path and data) is None, "Must provide either data_path or data, not both"

        data = torch.load(data_path) if data_path else data
        self.dataset = MotionDataset(data)
        super().__init__(self.dataset, **kwargs)

    @property
    def obs_dim(self):
        return self.dataset.data["observations"].shape[1]

    @property
    def data_keys(self):
        return self.dataset.data.keys()
