import numpy as np
import os
import itertools

import torch
from torch.utils.data import Dataset, DataLoader


class GetLuptitudes():
    """Transform for SCARLET image normalization"""

    def __init__(self, Q = 0.5, S = 2731):
        self.Q = Q
        self.S = S

    def __call__(self, imgs):
        imgs = np.arcsinh(self.Q * imgs / self.S) / self.Q
        return imgs


class ChannelNormalized():
    """Normalize each channel in the image"""
    def __call__(self, imgs):
        imgs /= imgs.sum(axis=(2, 3), keepdims=True)
        return imgs


class GetColors():
    """
    Computes pairwise difference (colors) between the channels
    By default computes n choose 2 number of colors where n is the number of input chanels
    You can pass custom_permuataions, which is an iterable of tuples of indices that
    will be used for color calculations.
    """
    def __init__(self, num_ch=6, custom_permutaions=None):
        self.num_ch = num_ch
        if custom_permutaions is None:
            self.permutations = itertools.combinations(range(self.num_ch), 2)

    def __call__(self, imgs):
        colors = []
        for (i, j) in self.permutations:
            color = imgs[:, i, :, :] - imgs[:, j, :, :]
            color = np.expand_dims(color, 1)
            colors.append(color)
        return np.concatenate(colors, axis=1)


class ConcatenateWithColors():
    def __call__(self, imgs):
        norm = ChannelNormalized()(imgs)
        clrs = GetColors()(norm)
        return np.concatenate([imgs, clrs], axis=1)


class BlendDataset(Dataset):
    def __init__(self, path, transform=None, test_run=False):

        files = os.listdir(path)
        self.paths = [os.path.join(path, file) for file in files if file.endswith('.npz')]
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
        data = np.load(self.paths[idx], allow_pickle=True)
        blend = data['imgs']
        num = data['nums']
        # apply transformations
        if self.transform:
            blend = self.transform(blend)
        # convert to tensor
        blend = torch.from_numpy(blend).float()
        num = torch.from_numpy(num)
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