from keras.layers import  Input, MaxPool2D, GlobalAveragePooling2D
from keras.layers import Concatenate
from models.basenet import BaseNet
from keras.models import Model

from layers.layer import conv2D_layer

class critic_2D_pse_h(BaseNet):
    def __init__(self, conf):
        super(critic_2D_pse_h, self).__init__(conf)

    def build(self, _subname = None):
        inp_shape         = self.conf.input_shape
        f                 = self.conf.filters
        name          = self.conf.name+_subname if _subname else self.conf.name

        d_input = Input(inp_shape)

        # (batch size, 160, 208, 1)
        conv1_1 = conv2D_layer(d_input, name="conv1_1", filters=f, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        pool1 = MaxPool2D()(conv1_1) # (batch size, 80, 104, filters)

        # (batch size, 80, 104, filters)
        conv2_1 = conv2D_layer(pool1, name="conv2_1", filters=f*2, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        pool2 = MaxPool2D()(conv2_1) # (batch size, 40, 52, filters*2)

        #  (batch size, 40, 52, filters*2)
        conv3_1 = conv2D_layer(pool2, name="conv3_1", filters=f*4, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        conv3_2 = conv2D_layer(conv3_1, name="conv3_2", filters=f*4, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        pool3 = MaxPool2D()(conv3_2) # (batch size, 20, 26, filters*4)

        # (batch size, 20, 26, filters*4)
        conv4_1 = conv2D_layer(pool3, name="conv4_1", filters=f*8, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        conv4_2 = conv2D_layer(conv4_1, name="conv4_2", filters=f*8, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal")
        pool4 = MaxPool2D()(conv4_2)
        # (batch size, 10, 13, filters*8)
        # =============================================================================================================

        mid1_1 = conv2D_layer(pool4, name="mid1_1", filters=f*8, kernel_size=3, strides=1, padding='same',
                                 activation='relu', kernel_initializer="he_normal")
        # (batch size, 10, 13, 32)
        mid1_2 = conv2D_layer(mid1_1, name="mid1_2", filters=f*8, kernel_size=3, strides=1, padding='same',
                                 activation='relu', kernel_initializer="he_normal")
        # (batch size, 10, 13, 32+f*8)
        mid_concat1 = Concatenate()([pool4, mid1_2])
        # =============================================================================================================
        conv5_1 = conv2D_layer(mid_concat1, name="conv5_1", filters=f*16, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal" )
        conv5_2 = conv2D_layer(conv5_1, name="conv5_2", filters=f*16, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal" )

        # (batch size, 10, 13, filters*16)
        convD_1 = conv2D_layer(conv5_2, name="convD_1", filters=f*16, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer="he_normal" )
        convD_2 = conv2D_layer(convD_1, name="convD_2", filters=1, kernel_size=3, strides=1, padding='same',
                                  activation='linear', kernel_initializer="he_normal")
        # (batch size, 10, 13, filters*16)
        averagePool = GlobalAveragePooling2D()(convD_2)

        self.model = Model([d_input], averagePool, name=name)