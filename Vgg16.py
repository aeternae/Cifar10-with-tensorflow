from __future__ import print_function
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras import regularizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import SGD
import os
import argparse
import random
import numpy as np
# from scipy.misc import imread, imresize, imsave
import pickle
import data_utility

from sklearn.model_selection import StratifiedKFold

# parser = argparse.ArgumentParser()
# parser.add_argument('--train_dir', default='./train/')
# parser.add_argument('--test_dir', default='./test/')
# parser.add_argument('--log_dir', default='./tf_dir')
# parser.add_argument('--batch_size', default=16)
# parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument("-o", "--output", required=False, type=str, help="path for output data")
# args = parser.parse_args()
# checkpoint_path = os.path.join('D:/VGG_for_cifar10/output', 'cp.ckpt')
# log_path = args.log_dir
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# type_list = ['cat', 'dog']


def vgg_16_net():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', name='conv1_block'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_block'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_block'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4_block'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5_block'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6_block'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv7_block'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv8_block'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv9_block'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv10_block'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv11_block'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv12_block'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv13_block'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # model.add(Dense(1, activation='sigmoid'))
    return model



def unpickle(file):
    with open(file, 'rb') as fo:
        file_dict = pickle.load(fo, encoding='bytes')
    return file_dict


def train(model):
    batch_size = 64
    num_classes = 10
    epochs = 100
    x_train, y_train, x_test, y_test = data_utility.prepare_data()

#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
#     y_train = utils.to_categorical(y_train, num_classes)
#     y_test = utils.to_categorical(y_test, num_classes)

    # sgd = SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    data_augmentation = True

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by dataset std
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in 0 to 180 degrees
            width_shift_range=0.1,  # randomly shift images horizontally
            height_shift_range=0.1,  # randomly shift images vertically
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_test, y_test),
            workers=4)

if __name__ == '__main__':
    try:
        model = vgg_16_net()
        train(model)
        # model.load_weights(args.log_dir + '/model.h5')
        # predict(model)
    except Exception  as err:
        print(err)
