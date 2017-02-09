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
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot
plt.style.use('bmh')

def plot_model(model, file_name):
    plot(model, to_file = file_name, show_shapes = True, show_layer_names = False)
    print("model graph is saved in %s", file_name)

def plot_dis(series, file_name, bins = 80):
    plt.figure()
    series.hist(bins = bins)
    plt.savefig(file_name)
    plt.show()
