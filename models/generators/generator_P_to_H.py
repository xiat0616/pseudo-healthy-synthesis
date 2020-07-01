from keras.layers import Conv2D, Activation, Input, Concatenate
from models.basenet import BaseNet
from keras.models import Model
from layers.layer import residual_block_bn, conv2D_layer_bn, g_downsampling_bn
from layers.layer import  g_upsampling_bn
from easydict import EasyDict
from configuration.exp_supervised import EXPERIMENT_PARAMS

class generator_P_to_H(BaseNet):
    def __init__(self, conf):
        super(generator_P_to_H, self).__init__(conf)

    def build(self, _subname=None):
        f        = self.conf.filters
        name     = self.conf.name + _subname if _subname else self.conf.name
        g_input  = Input(shape=self.conf.input_shape)

        # (batch_size, 208, 160, 1)
        conv1_1 = conv2D_layer_bn(g_input, name=None, filters=32, kernel_size=7, strides=1, padding='same',
                                  activation='relu')
        conv1_1 = conv2D_layer_bn(conv1_1, name=None, filters=32, kernel_size=7, strides=1, padding='same',
                                  activation='relu')
        # (batch_size, 208, 160, 32)
        conv2_1 = g_downsampling_bn(conv1_1, filters=f*2)
        # (batch_size, 104, 80, 64)
        conv3_1 = g_downsampling_bn(conv2_1, filters=f*4)
        # (batch_size, 52, 40, 128)
        # ========== residual part ====================================
        res_1_1 = residual_block_bn(conv3_1, f*4)
        # (batch_size, 52, 40, 128)
        res_1_2 = residual_block_bn(res_1_1, f*4)
        # (batch_size, 52, 40, 128)
        res_1_3 = residual_block_bn(res_1_2, f*4)
        # (batch_size, 52, 40, 128)
        res_1_4 = residual_block_bn(res_1_3, f*4)
        # (batch_size, 52, 40, 128)
        res_1_5 = residual_block_bn(res_1_4, f*4)
        # (batch_size, 52, 40, 128)
        res_1_6 = residual_block_bn(res_1_5, f*4)
        # (batch_size, 52, 40, 128)
        # ========== residual part end ================================
        conv4_1 = g_upsampling_bn(res_1_6, f*4)
        # (batch_size, 104, 80, 128)
        concat_1 = Concatenate()([conv4_1, conv2_1])
        # (batch_size, 104, 80, 192)
        conv4_2 = conv2D_layer_bn(concat_1, name="conv4_2", filters=f*2, kernel_size=3, strides=1, padding='same',
                                  activation='relu')
        # (batch_size, 104, 80, 64)
        conv5_1 = g_upsampling_bn(conv4_2, f*2)
        # (batch_size, 208, 160, 64)
        concat_2 = Concatenate()([conv5_1, conv1_1])
        # (batch_size, 208, 160, 96)
        conv5_2 = conv2D_layer_bn(concat_2, name="conv5_1", filters=f, kernel_size=3, strides=1, padding='same',
                                  activation='relu')
        # (batch_size, 208, 160, 32)
        conv5_3 = Conv2D(filters=1, name="conv5_3", kernel_size=7, strides=1, padding="same", activation="relu")(conv5_2)
        # (batch_size, 208, 160, 1)
        g_output = Activation(self.conf.G_activation)(conv5_3)
        # (batch_size, 208, 160, 1)

        self.model =  Model(inputs=g_input, outputs=g_output, name=name)
        self.model.summary()

if __name__=='__main__':
    conf = EasyDict(EXPERIMENT_PARAMS)
    midl_baseline_w_gan = generator_P_to_H(conf.gen_params_p_to_h)
    midl_baseline_w_gan.build()