from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose, LeakyReLU
from keras.layers import UpSampling2D, Dropout, BatchNormalization
import numpy as np
import utils
from keras.initializers import RandomNormal
from configuration.exp_supervised import EXPERIMENT_PARAMS
from easydict import EasyDict
import keras.backend as K
from models.basenet import BaseNet
from keras.optimizers import Adam
from utils.cost import dice_coef_loss
'''
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upconv: use transposed conv instead of upsamping + conv if false
residual: add residual connections around each conv block if true
'''


def dice_loss_for_testing(y_pred, y_true, smooth =1):
    y_int = y_true * y_pred
    # Technically this is the negative of the Sorensen-Dice index. This is done for minimization purposes
    dice_score= (2 * np.sum(np.abs(y_int))+smooth) / (np.sum(np.abs(y_true)) +smooth+ np.sum(np.abs(y_pred)))
    return 1 - dice_score

def mixed_loss(y_pred, y_true):
    return K.binary_crossentropy(y_pred, y_true)+utils.dice_coef_loss(y_pred, y_true)

def conv_block(m, dim, acti, bn, res, do=0):
    n= Conv2D(dim, 3, activation=acti, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(m)
    n= BatchNormalization()(n) if bn else n
    n= Dropout(do)(n) if do else n
    n= Conv2D(dim, 3, activation=acti, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(n)
    n= BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
         n = conv_block(m, dim, acti, bn, res)
         m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
         m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
         if up:
             m = UpSampling2D()(m)
             m = Conv2D(dim, 2, activation=acti, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02) )(m)
             m = LeakyReLU(0.1)(m)
         else:
             m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(m)
             m = LeakyReLU(0.1)(m)
         n = Concatenate()([n, m])
         m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m

def Unet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='linear',
		 dropout=0.5, batchnorm=True, maxpool=True, upconv=True, residual=True, name=None):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o, name=name)

class mask_segmentor(BaseNet):
    """
    The segmentor of MIDL
    """
    def __init__(self, conf):
        super(mask_segmentor, self).__init__(conf)
        self.conf = conf

    def build(self, _subname=None):
        f        = self.conf.filters
        name     = self.conf.name + _subname if _subname else self.conf.name
        # (batch_size, 208, 160, 1)
        self.model = Unet(self.conf.input_shape, out_ch=1, start_ch=f, depth=4, inc_rate=2., activation='relu',
		                  dropout=0.3, batchnorm=True, maxpool=True, upconv=True, residual=True, name=name)

        try:
            self.model.compile(Adam(lr=self.conf.lr, beta_1=self.conf.beta1, decay=self.conf.decay), loss=dice_coef_loss)
        except:
            print("Unet not compiled")
        self.model.summary()


if __name__=='__main__':
    midl_segmentor=mask_segmentor(conf=EasyDict(EXPERIMENT_PARAMS).seg_params)
    midl_segmentor.build()
    midl_segmentor.model.summary()