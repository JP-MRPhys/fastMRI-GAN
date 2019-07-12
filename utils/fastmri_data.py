import h5py
import numpy as np

import pathlib
import random
from matplotlib import pyplot as plt
from utils.subsample import MaskFunc
import utils.transforms as T


def get_training_pair(file, centre_fraction, acceleration):


    hf = h5py.File(file)

    volume_kspace = hf['kspace'][()]
    volume_image = hf['reconstruction_esc'][()]
    mask_func = MaskFunc(center_fractions=[centre_fraction], accelerations=[acceleration])  # Create the mask function object
    volume_kspace_tensor = T.to_tensor(volume_kspace)
    masked_kspace, mask = T.apply_mask(volume_kspace_tensor, mask_func)
    masked_kspace_np=masked_kspace.numpy().reshape(masked_kspace.shape)

    return np.expand_dims(volume_image,3), masked_kspace_np


def get_random_accelerations():
   """
   obtain random centre_fractions and accelerations between 1 and 15
   :return:
   """
   acceleration = np.random.randint(1, high=15, size=1)
   centre_fraction = np.random.uniform(0, 1, 1)
   decimal = np.random.randint(1, high=3, size=1)
   centre_fraction = centre_fraction / (10 ** decimal)
   return float(centre_fraction), float(acceleration)


def train(datadir):

    files = list(pathlib.Path(datadir).iterdir())
    random.shuffle(files)
    for file in files:

        centre_fraction,acceleration=get_random_accelerations()
        image, masked_kspace=get_training_pair(file, centre_fraction=centre_fraction,acceleration=acceleration)
        print(image.shape)
        print(masked_kspace.shape)


if __name__ == '__main__':

    training_datadir='/media/jehill/Data/ML_data/fastmri/singlecoil/train/singlecoil_train/'
    train(training_datadir)
