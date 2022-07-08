import numpy as np
import os
import time

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
        save_path = os.path.join(DATA.dataset_location, 'NumberedGalaxyBlends', key)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        t_start = time.time()
        t_stemp = t_start
        # iterate and create data files
        for i in range(sample_size[key]):
            batch = next(gen)
            imgs = batch['blend_images']
            isolated = batch['isolated_images']
            meta = batch['blend_list']
            numGal = np.array([len(table) for table in meta])
            filename = f'{str(i).zfill(9)}.npz'
            # add the blends with 0 galaxies by randomly erasing galaxy fluxes
            idx_0_galaxy = np.random.rand(imgs.shape[0]) < 1/(DATA.max_number + 1)
            imgs[idx_0_galaxy] = imgs[idx_0_galaxy] - isolated[idx_0_galaxy].sum(axis=1)
            numGal[idx_0_galaxy] = 0
            np.savez(os.path.join(save_path, filename), imgs=imgs, nums=numGal)
            # print progress (we can't use tqdm because of BTK unfortunately)
            if (i + 1) % 100 == 0:
                t_stemp = time.time() - t_stemp
                t_overall = time.time() - t_start
                print(f'[{i+1}/{sample_size[key]}] images for {key}, time/100 {t_stemp:.3f}, time overall {t_overall:.3f}')




if __name__ == "__main__":
    print(f"Saving data to: {DATA.dataset_location}")
    print("Starting generation of blends:")
    print(f"# TRAIN images: {DATA.train_batches * DATA.file_batch}")
    print(f"# VALID images: {DATA.valid_batches * DATA.file_batch}")
    gen = get_generator()
    save_data(gen)
