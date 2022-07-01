import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader


class NormalizeSCARLET():
    """Transform for SCARLET image normalization"""

    def __init__(self, Q, S):
        self.Q = Q
        self.S = S

    def __call__(self, batch):
        batch = np.arcsinh(self.Q * batch / self.S) / self.Q
        return batch


class BlendDataset(Dataset):
    def __init__(self, path, transform=None, test_run=False):

        files = os.listdir(path)
        self.paths = [os.path.join(path, file) for file in files]
        self.transform = transform
        try:
            self.file_batch = np.load(self.paths[0])['imgs'].shape[0]
        except:
            raise Exception("Could not read data. Check if folder has at least one data sample")

        # for testing purposes
        if test_run:
            self.paths = self.paths[0:50]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        blend = data['imgs']
        num = data['nums']
        # convert to tensor
        blend = torch.from_numpy(blend).float()
        num = torch.from_numpy(num) - 1  # shift labels by 1 (because of CELoss convention)
        if self.transform:
            blend = self.transform(blend)
        return blend, num


def collate_fn(data):
    blends, nums = zip(*data)
    return torch.cat(blends), torch.cat(nums)


def get_dataloaders(train_path, test_path, batch_size, transforms, test_run=False):
    train_data = BlendDataset(train_path, transforms, test_run)
    valid_data = BlendDataset(test_path, transforms)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size//train_data.file_batch,
        collate_fn=collate_fn,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size//valid_data.file_batch,
        collate_fn=collate_fn,
        shuffle=False
    )
    return train_loader, valid_loader