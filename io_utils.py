"""
Behavioral Cloning Agent Using Keras

"""
#
# Peter Lai
# Behavioral Cloning (steering)
# version 1.0
# Simulator Powered by Udacity
#

import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from vis_utils import plot_dis

class gen_from_file:
    """
    a class that makes two generator for training and validating
    """

    def __init__(self, file_names, im_shape, val_rate = 0.15, view = "all", keep_straight = 0.1, resample = True):
        """
        params:
            file_names: driving_log.csv files
            im_shape: tuple (width, height)
            val_rate: validation_data propotion
            view: using all view or just center
        """

        # read the csv (with image path data) but not image data
        col_names = np.array(['center','left', 'right', 'steering'])
        self.im_shape = im_shape
        self.data = None
        for f in file_names:
            if self.data is None:
                self.data = pd.read_csv(f, names = col_names, usecols = range(0, 4))
            else:
                df = pd.read_csv(f, names = col_names, usecols = range(0, 4))
                self.data = pd.concat([self.data, df])

        # reduce the 0 steering samples
        if resample:
            self.resample(keep_straight)
        plot_dis(self.data.iloc[:,3], 'dis.png')

        # make random tuples for generating [data index, view]
        size = self.data.shape[0]
        random_tup = None
        if view is 'all':
            random_tup = np.vstack([np.repeat(np.arange(size), 3),
                                    np.tile(np.arange(3), size)]).T
        elif view is 'center':
            random_tup = np.vstack([np.arange(size), [0]*size]).T
        np.random.shuffle(random_tup)
        tup_size = random_tup.shape[0]

        # train val split
        val_sep = int(val_rate*tup_size)
        self.train_tup = random_tup[:-val_sep]
        self.val_tup = random_tup[-val_sep:]
        self.train_size = self.train_tup.shape[0]
        self.val_size = self.val_tup.shape[0]


    def resample(self, keep_size = 0.1):
        """
        reduce the no steering data
        colunm 4 should be the steering angle
        """
        straight = self.data[self.data.iloc[:,3] == 0.0]
        straight = straight.sample(frac = keep_size)
        self.data = pd.concat([self.data[self.data.iloc[:,3] != 0.0], straight])



    def normalize(self, images, a = -.5, b = .5):
        """
        assuming images is numpy array
        """
        color_max = 255
        return a + images * (b - a) / color_max



    def get_data(self, tup):
        """
        from indexing to images data in numpy array
        """
        X = []
        for k, v in tup:
            f = self.data.iloc[k,v].strip()
            img = cv2.resize(np.asarray(Image.open(f)), self.im_shape)
            X.append(img.tolist())
        y = self.data.iloc[:,3].as_matrix()[tup.T[0]]
        return self.normalize(np.array(X)), y






    def gen(self, tup, batch_size = 64):
        """
        param:
            batch_size: the batch_size for training
            file_name: a csv file match the ouput format of the sample file

        yield:
            data of float32, shape = [batch_size, imsize, imsize, n_channels]
        """
        tup_size = tup.shape[0]
        # generate the data
        i = 0
        while True:
            end = i + batch_size
            batch_tup = tup[i : end]
            i = end if end < tup_size - 1 else 0

            yield self.get_data(batch_tup)


    def train_gen(self, batch_size = 64):
        return self.gen(self.train_tup, batch_size)

    def val_gen(self, batch_size  = 64):
        return self.gen(self.val_tup, batch_size)
