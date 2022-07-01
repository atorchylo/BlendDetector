import numpy as np
import os

import btk.survey
import btk.draw_blends
import btk.catalog
import btk.sampling_functions

from config import DATA

# Silence TQDM progress bar
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def get_generator():
    """Generates BTK generator for drawing galaxy blends"""
    catalog = btk.catalog.CatsimCatalog.from_file(DATA.catalog_path)
    lsst = btk.survey.get_surveys('LSST')
    sampler = btk.sampling_functions.DefaultSampling(
        max_number=DATA.max_number,
        stamp_size=DATA.stamp_size,
        maxshift=DATA.max_shift
    )
    generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampler,
        lsst,
        batch_size=DATA.file_batch,
        stamp_size=DATA.stamp_size,
        cpus=DATA.number_cpu,
        add_noise='all',
        verbose=False
    )
    return generator


def save_data(gen):
    """
    Saves generated images and corresponding number of galaxies locally
    """
    sample_size = {'train': DATA.train_batches, 'valid': DATA.valid_batches}
    for key in sample_size:
        # create folders
        save_path = os.path.join('data', 'NumberedGalaxyBlends', key)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # iterate and create data files
        for i in range(sample_size[key]):
            batch = next(gen)
            imgs = batch['blend_images']
            meta = batch['blend_list']
            numGal = np.array([len(table) for table in meta])
            filename = f'{str(i).zfill(9)}.npz'
            np.savez(os.path.join(save_path, filename), imgs=imgs, nums=numGal)


if __name__ == "__main__":
    print("Starting generation of blends:")
    print(f"# TRAIN images: {DATA.train_batches * DATA.file_batch}")
    print(f"# VALID images: {DATA.valid_batches * DATA.file_batch}")
    gen = get_generator()
    save_data(gen)