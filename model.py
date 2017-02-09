"""
Behavioral Cloning Agent Using Keras

"""
#
# Peter Lai
# Behavioral Cloning (steering)
# version 1.0
# Simulator Powered by Udacity
#

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, BatchNormalization, Lambda
from keras.optimizers import Adam
from keras.applications import VGG16
from io_utils import gen_from_file
from vis_utils import plot_model
import os, errno
# define images for training
imsize = 64
n_channels = 3
view = 'all'
batch_size = 128
epochs = 8
mod = 'PNET'

# Define the architecture
def PNet():
    """
    a keras model definition
    """

    lin = Input(shape = (imsize, imsize, n_channels))
    l = Convolution2D(16, 7, 7, subsample=(1, 1), border_mode = 'same', activation = 'relu')(lin)
    l = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode = 'same', activation = 'relu')(l)
    l = Dropout(0.8)(l)
    l = Convolution2D(128, 5, 5, subsample=(2, 2), border_mode = 'same', activation = 'relu')(l)
    l = Flatten()(l)
    l = Dense(512, activation = 'relu')(l)
    l = Dropout(0.8)(l)
    l = Dense(32, activation = 'relu')(l)
    lout = Dense(1)(l)
    model = Model(input = lin, output = lout)
    model.compile(optimizer = 'adam', loss = 'mse')
    return model


def VGG_reg(bp_stop = -5):
    """
    feature extraction using vgg and then do regression
    """
    lin = Input(shape = (imsize, imsize, 3))
    net = VGG16(include_top = False, weights='imagenet', input_tensor = lin)
    lm = net.output
    l = Flatten()(lm)
    l = Dense(1024, activation = 'relu')(l)
    l = Dense(256, activation = 'relu')(l)
    l = Dropout(0.6)(l)
    l = Dense(32, activation = 'relu')(l)
    lout = Dense(1)(l)
    model = Model(input = net.input, output = lout)
    for l in model.layers[bp_stop:]:
        l.trainable = True
    for l in model.layers[:bp_stop]:
        l.trainable = False
    model.compile(optimizer = 'adam', loss = 'mse')
    return model


#def NvdApr():
#    """
#    WARNING: Gradient Vanishing
#    """
#    lin = Input(shape = (imsize, imsize, 3))
#    l = Convolution2D(24, 5, 5, subsample=(2, 2), border_mode = 'same', activation = 'relu')(lin)
#    l = Convolution2D(36, 5, 5, subsample=(2, 2), border_mode = 'same', activation = 'relu')(l)
#    l = Convolution2D(48, 5, 5, subsample=(2, 2), border_mode = 'same', activation = 'relu')(l)
#    l = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode = 'same', activation = 'relu')(l)
#    l = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode = 'same', activation = 'relu')(l)
#    l = Flatten()(l)
#    l = Dense(1164, activation = 'relu')(l)
#    l = Dense(100, activation = 'relu')(l)
#    l = Dense(50, activation = 'relu')(l)
#    l = Dense(10, activation = 'relu')(l)
#    lout = Dense(1, activation = 'relu')(l)
#    model = Model(input = lin, output = lout)
#    model.compile(optimizer = Adam(1e-6), loss = 'mse')
#    return model

def select_model(net = 'PNET'):
    print("%s is selected"%net)
    if net == 'PNET':
        return PNet()
    if net == 'VGG':
        return VGG_reg()
    #if net == 'NVD':
        #return NvdApr()

# define the model
model = select_model(mod)
plot_model(model, ('model%s.png'%mod))
model.summary()
#
# NOTE: Feel free to replace it with your own data
#
files = ['./recv/driving_log.csv',
         './full_speed_track_1/driving_log.csv',
         './full_speed_track_2/driving_log.csv']

# create the generate from file object
gff = gen_from_file(files[:3], (imsize, imsize), view = view, resample = True)
train_size, val_size = gff.train_size, gff.val_size
print("train_size : %d" %train_size)
print("val_size : %d"%val_size)

# train the model
model.fit_generator(gff.train_gen(batch_size = batch_size),
                    samples_per_epoch = train_size,
                    nb_epoch = epochs,
                    validation_data = gff.val_gen(batch_size = batch_size),
                    nb_val_samples = val_size)

# Save the model and weights
with open(('model%s.json'%mod), 'w') as f:
    f.write(model.to_json())
try:
    os.remove(('model%s.h5'%mod))
except OSError:
    pass
model.save_weights(('model%s.h5'%mod), overwrite = True)
