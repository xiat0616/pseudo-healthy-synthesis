#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: tian
"""
from keras.layers import BatchNormalization
from keras.layers import Activation, UpSampling2D, LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import Add

from keras.initializers import RandomNormal

def residual_block(l0, filters):
    l = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)
               )(l0)
    l = Activation('relu')(l)

    l = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(l)
    l = Activation('relu')(l)
    return Add()([l0, l])

def residual_block_bn(l0, filters):
    l = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(l0)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)

    l = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    return Add()([l0, l])

def g_upsampling_bn(l0, filters):
    l = UpSampling2D(size=2)(l0)
    l = Conv2D(filters=filters, kernel_size=3,  strides=1, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)

    return l

def g_downsampling_bn(l0, filters, kernal_size=3):
    l = Conv2D(filters, kernel_size=kernal_size, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(l0)
    l = BatchNormalization()(l)
    l =Activation('relu')(l)

    l = Conv2D(filters, kernel_size=kernal_size, strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(l)
    l = BatchNormalization()(l)
    l =Activation('relu')(l)
    return l

def d_downsampling_bn(l0, filters):
    l = Conv2D(filters=filters, kernel_size=4, strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(l0)
    l = BatchNormalization()(l)
    l = LeakyReLU(0.2)(l)
    return l

def conv2D_layer_bn(l0, name=None, filters=32, kernel_size=3, strides=1, padding='same', activation='relu'):
    l = Conv2D( filters=filters, name=name, kernel_size=kernel_size, strides=strides, padding=padding,
                activation=activation,kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(l0)
    l = BatchNormalization()(l)
    return LeakyReLU(0.2)(l)


def conv2D_layer(l0, name=None, filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                    kernel_initializer="he_normal"):
    l = Conv2D( filters=filters, name=name, kernel_size=kernel_size, strides=strides, padding=padding,
                activation=activation, kernel_initializer=kernel_initializer)(l0)
    # l = BatchNormalization()(l)
    return LeakyReLU(0.2)(l)


def deconv2D_layer_bn(l0, name=None, filters=32, kernel_size=3, strides=2, padding='same', activation='relu',
                      kernel_initializer="he_normal"):
    l = Conv2DTranspose( filters=filters, name=name, kernel_size=kernel_size, strides=strides, padding=padding,
                activation=activation, kernel_initializer=kernel_initializer)(l0)
    l = BatchNormalization()(l)
    return l

def conv2D_layer(l0, name=None, filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                    kernel_initializer="he_normal"):
    l = Conv2D( filters=filters, name=name, kernel_size=kernel_size, strides=strides, padding=padding,
                activation=activation, kernel_initializer=kernel_initializer)(l0)
    return l



